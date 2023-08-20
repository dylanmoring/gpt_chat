from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio

from throttler import Throttler

import openai
import tiktoken
import backoff

from ..metaclasses import SelfLogging

cost_map = {
    # Structure is model: object_type: usage_key: rate
    # Note: If rate granularity ever exceeds 0.00001, rounding step should be extended past 8 decimals
    'gpt-3.5-turbo-16k': {
        'chat.completion': {
            'prompt_tokens': 0.003 / 1000,
            'completion_tokens': 0.004 / 1000,
        }
    },
    'gpt-3.5-turbo': {
        'chat.completion': {
            'prompt_tokens': 0.0015 / 1000,
            'completion_tokens': 0.002 / 1000,
        }
    },
    'gpt-4-32k': {
        'chat.completion': {
            'prompt_tokens': 0.06 / 1000,
            'completion_tokens': 0.12 / 1000,
        }
    },
    'gpt-4': {
        'chat.completion': {
            'prompt_tokens': 0.03 / 1000,
            'completion_tokens': 0.06 / 1000,
        }
    },
    'text-embedding-ada-002': {
        'list': {  # List of embeddings
            'prompt_tokens': 0.0004 / 1000
        }
    }
}

token_limit_map = {
    # Structure is model: object_type: usage_key: rate
    'gpt-3.5-turbo-16k': {
        'chat.completion': 16000
    },
    'gpt-3.5-turbo': {
        'chat.completion': 4000
    },
    'gpt-4-32k': {
        'chat.completion': 32000
    },
    'gpt-4': {
        'chat.completion': 8000
    },
    'text-embedding-ada-002': {
        'list': 8000  # List of embeddings
    }
}


class OpenAIToolException(Exception):
    pass


class TooManyTokens(OpenAIToolException):
    def __init__(self, token_length, token_limit):
        super().__init__(
            f'Request to OpenAI API is too large! Request was {token_length:,} tokens long and the limit is '
            f'{token_limit:,}'
        )


class OpenAIClient(metaclass=SelfLogging):
    embeddings_throttler = Throttler(rate_limit=3500, period=60)
    chat_throttler = Throttler(rate_limit=3500, period=60)
    edit_throttler = Throttler(rate_limit=20, period=60)
    embedding_model = "text-embedding-ada-002"
    chat_model = "gpt-4"
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    encoding = tiktoken.get_encoding("cl100k_base")

    def __init__(self, open_ai_api_key: Optional[str] = None):
        if open_ai_api_key is None and openai.api_key is None:
            raise ValueError(
                'An API key is required to instantiate this client. Supply one in the init call or set one globally.')
        elif open_ai_api_key is None:
            pass
        else:
            openai.api_key = open_ai_api_key

    @classmethod
    def get_token_length(cls, text: str) -> int:
        return len(cls.encoding.encode(text))

    @classmethod
    def tokenize(cls, text: str):
        return cls.encoding.encode(text)

    @classmethod
    def get_token_limit(cls, model: str, request_type: str = 'chat.completion') -> int:
        model = cls.chat_model if model is None else model
        # Get cost map
        best_key = [k for k in token_limit_map if k in model][0]
        return token_limit_map[best_key][request_type]

    @classmethod
    def check_request_size(cls, model: str, request_type: str, request: str) -> bool:
        token_limit = cls.get_token_limit(model, request_type)
        token_length = cls.get_token_length(request)
        if token_length > token_limit:
            raise TooManyTokens(token_length, token_limit)
        return True

    @classmethod
    def get_consumption(cls, response: dict) -> Tuple[int, float]:
        usage = response['usage']
        model = response['model']
        object_type = response['object']
        # Get cost map
        best_key = [k for k in cost_map if k in model][0]
        cost = cost_map[best_key][object_type]
        # Get total cost
        total_cost = 0
        total_tokens = 0
        for token_type, tokens in usage.items():
            if token_type in cost:
                total_cost += round(cost[token_type] * tokens, 8)
                total_tokens += tokens
        return total_tokens, total_cost

    @classmethod
    def log_consumption(cls, response: dict):
        total_tokens, total_cost = cls.get_consumption(response)
        cls.log.info(
            f'Processed {total_tokens:,} tokens for ${total_cost:.3f}'
        )

    @classmethod
    async def get_embedding(cls, text: str) -> Optional[List[float]]:
        # Check that text isn't too long
        token_length = cls.get_token_length(text)
        if token_length > cls.get_token_limit(cls.embedding_model, 'list'):
            cls.log.warning(
                f"Couldn't generate embedding because the input text was too long! It was {token_length:,} tokens long."
            )
            return None
        if token_length > cls.max_tokens:
            cls.log.warning(
                f"Couldn't generate embedding because the input text was too long! It was {token_length:,} tokens long."
            )
            return None
        # replace newlines, which can negatively affect performance.
        text = text.replace('\n', '')
        async with cls.embeddings_throttler:
            response = await openai.Embedding.acreate(
                input=text,
                model=cls.embedding_model
            )
            cls.log_consumption(response)
            embedding = response["data"][0]["embedding"]
            return embedding

    @classmethod
    async def get_embeddings(cls, list_of_text: List[str]) -> List[List[float]]:
        assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."
        text_s = list_of_text.copy()
        # replace newlines, which can negatively affect performance.
        text_s = [t.replace('\n', '') for t in text_s]
        token_length = [cls.get_token_length(t) for t in text_s]
        if max(token_length) > cls.get_token_limit(cls.embedding_model, 'list'):
            cls.log.warning(
                f'Couldn\'t generate embedding because the input text was too long! The longest was '
                f'{max(token_length):,} tokens long.'
            )
            return []
        async with cls.embeddings_throttler:
            response = await openai.Embedding.acreate(
                input=list_of_text,
                model=cls.embedding_model
            )
        cls.log_consumption(response)
        embeddings = sorted(response, key=lambda x: x["index"])  # maintain the same order as input.
        embeddings = [d["embedding"] for d in embeddings]
        return embeddings

    @classmethod
    async def get_embeddings_column(cls, text_column):
        raise NotImplementedError('To use get_embeddings_column, pandas must be installed.')

    @classmethod
    async def create_completion(
            cls,
            prompt: str,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=64
    ):
        async with cls.embeddings_throttler:
            ai_response = await openai.Completion.acreate(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens
            )
        ai_response = ai_response['choices'][0]['text']
        return ai_response.strip()

    @classmethod
    @backoff.on_exception(
        backoff.constant,
        openai.error.RateLimitError,
        jitter=lambda: backoff.full_jitter(15),
        max_tries=14,  # With the given parameters, 5 minutes would roughly allow for 12 tries.
        interval=15,
        giveup=lambda e: 'overloaded' not in str(e) and 'Rate limit reached' not in str(e)
    )
    @backoff.on_exception(
        backoff.expo,
        openai.error.APIError,
        max_time=600,
        giveup=lambda e: 'Gateway timeout' not in str(e) and 'Bad gateway' not in str(e)
    )
    @backoff.on_exception(
        backoff.expo,
        asyncio.TimeoutError,
        max_tries=2
    )
    async def get_chat(
            cls,
            messages: List[Dict[str, str]],
            model: str = None,
            temperature: int = 1,
            top_p: int = 1,
            n: int = 1,
            stop: Optional[str] = None,
            max_tokens: Optional[int] = None,
            presence_penalty: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            logit_bias: Optional[dict] = None,
            user: str = None
    ):
        model = cls.chat_model if model is None else model
        # Remove empty values from arguments
        kwargs = {
            'messages': messages,
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'n': n,
            'stop': stop,
            'max_tokens': max_tokens,
            'presence_penalty': presence_penalty,
            'frequency_penalty': frequency_penalty,
            'logit_bias': logit_bias,
            'user': user
        }
        kwargs = {a: v for a, v in kwargs.items() if v is not None}
        cls.check_request_size(model, 'chat.completion', '\n'.join(m['content'] for m in messages))
        async with cls.chat_throttler:
            ai_response = await asyncio.wait_for(openai.ChatCompletion.acreate(**kwargs), 390)  # Timeout after 5 mins
        cls.log_consumption(ai_response)
        return ai_response

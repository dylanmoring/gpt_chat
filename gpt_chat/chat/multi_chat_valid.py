from typing import List, Union, Optional, Callable

from .chat import Chat
from .messages import Message, AssistantMessage


class MultiValidChat(Chat):
    def __init__(
            self,
            messages: List[Message],
            validation_function: Callable = lambda x: True,
            model: str = 'gpt-4',
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
        self.kwargs = locals()
        del self.kwargs['__class__']
        del self.kwargs['self']
        del self.kwargs['messages']
        del self.kwargs['validation_function']
        self.validation_function = validation_function
        super().__init__(*messages)

    async def get_next_message(self, return_original=False, **kwargs) -> Union[Message, dict]:
        # Get stored kwargs
        kwargs_to_use = self.kwargs.copy()
        kwargs_to_use.update(kwargs)
        kwargs_to_use['return_original'] = return_original
        response = await super().get_next_message(**kwargs_to_use)
        if not return_original:
            possible_messages = response
            for i, message in enumerate(possible_messages):
                if self.validation_function(message):
                    self.log.info(f'Returning message #{i+1} of {len(possible_messages)} possibilities.')
                    message = AssistantMessage(message)
                    self.append(message)  # Add message to chat if only one
                    return message
            self.log.error(f'Last Message:\n\n{message}')
            raise ValueError('No versions passed validation.')
        else:
            good_message = None
            for i, message in enumerate(response['choices']):
                message_text = message['message']['content']
                if self.validation_function(message_text):
                    self.log.info(f'Returning message #{i+1} of {len(response["choices"])} possibilities.')
                    good_message = message
                    break
            if good_message is not None:
                fake_original = response.copy()
                fake_original['choices'] = [good_message]
                return fake_original
            self.log.error(f'Last Message:\n\n{message}')
            raise ValueError('No versions passed validation.')

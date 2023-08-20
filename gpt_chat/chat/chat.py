from typing import List, Union
import logging

from ..metaclasses import SelfLogging

from ..client.open_ai_client import OpenAIClient
from .messages import Message, AssistantMessage


class Chat(list, metaclass=SelfLogging):
    client = OpenAIClient
    log_level = logging.INFO

    def __init__(
            self,
            *messages: Union[Message, List[Message]],
    ):
        if messages and isinstance(messages[0], list):
            super().__init__(messages[0])
        super().__init__(messages)
        self.cost = 0

    def __copy__(self):
        return Chat(*self)

    @property
    def api_repr(self) -> List[dict]:
        api_repr = [
            {
                'role': message.role,
                'content': str(message),
                'name': message.name,
            } for message in self
        ]
        # Remove empty names
        for message in api_repr:
            if not message['name']:
                del message['name']
        return api_repr

    @property
    def request_size(self):
        request_payload = '\n'.join([str(message) for message in self])
        token_length = self.client.get_token_length(request_payload)
        return token_length

    def check_request_size(self, response_allowance: int = 1000, model: str = None) -> bool:
        model = model if model else self.client.chat_model

        token_limit = self.client.get_token_limit(model, 'chat.completion') - response_allowance
        token_length = self.request_size
        if token_length > token_limit:
            self.log.debug(f'{token_length} total tokens requested out of {token_limit} allowed.')
            return False
        return True

    async def get_next_message(self, return_original: bool = False, **kwargs) -> Union[List[Message], Message, dict]:
        self.log_content(self.log_level)
        # Get the next message
        response = await self.client.get_chat(
            messages=self.api_repr,
            **kwargs
        )
        self.cost += self.client.get_consumption(response)[1]  # Get the second item in the output, which is cost
        output_messages = []
        for i, choice in enumerate(response['choices']):
            choice_message = Message(**choice['message'])
            self.log.log(
                level=self.log_level,
                msg=f'GPT Response Option {i}:\n\n{choice_message.log_formatted}'
            )
            output_messages.append(choice_message)
        if return_original:
            return response
        elif len(output_messages) == 1:
            message = AssistantMessage(output_messages[0])
            self.append(message)  # Add message to chat if only one
            return message
        else:
            return output_messages

    def log_content(self, level: logging.INFO):
        log_message = '\n\n'.join([message.log_formatted for message in self])
        self.log.log(
            level=level,
            msg=log_message
        )

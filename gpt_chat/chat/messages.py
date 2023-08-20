from typing import Optional


class Message(str):
    def __new__(
            cls,
            role: str,
            content: str,
            name: Optional[str] = None,
    ):
        instance = super().__new__(cls, content)
        instance.role = role.lower()
        instance.name = name
        return instance

    def __init__(
            self,
            role: str,
            content: str,
            name: Optional[str] = None,
    ):
        if role not in ('system', 'user', 'assistant'):
            raise ValueError(f'Invalid role: {role}')

    def __repr__(self) -> str:
        return f"{self.role.title()} Message: '{self}'"

    def __str__(self) -> str:
        return super().__str__()

    @property
    def log_formatted(self) -> str:
        msg_type = f'{self.role.title()} Message'
        if self.name:
            msg_type = f'{msg_type} ({self.name})'
        content = str(self)
        return f'{msg_type}:\n{content}'


class SystemMessage(Message):
    def __new__(
            cls,
            content: str,
            name: Optional[str] = None,
    ):
        instance = super().__new__(
            cls,
            role='system',
            content=content,
            name=name,
            )
        return instance

    def __init__(
            self,
            content: str,
            name: Optional[str] = None,
    ):
        super().__init__(
            role='system',
            content=content,
            name=name,
        )


class UserMessage(Message):
    def __new__(
            cls,
            content: str,
            name: Optional[str] = None,
    ):
        instance = super().__new__(
            cls,
            role='user',
            content=content,
            name=name,
            )
        return instance

    def __init__(
            self,
            content: str,
            name: Optional[str] = None,
    ):
        super().__init__(
            role='user',
            content=content,
            name=name,
        )


class AssistantMessage(Message):
    def __new__(
            cls,
            content: str,
            name: Optional[str] = None,
    ):
        instance = super().__new__(
            cls,
            role='assistant',
            content=content,
            name=name,
            )
        return instance

    def __init__(
            self,
            content: str,
            name: Optional[str] = None,
    ):
        super().__init__(
            role='assistant',
            content=content,
            name=name,
        )
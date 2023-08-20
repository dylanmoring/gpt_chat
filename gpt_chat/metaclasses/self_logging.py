from logging import getLogger


class SelfLogging(type):
    def __new__(mcs, name, bases, dict):
        x = super().__new__(mcs, name, bases, dict)
        x.log = getLogger(x.__name__)
        return x

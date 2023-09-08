from typing import List
import abc


class BaseCalculator(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def exec(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def get_state(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

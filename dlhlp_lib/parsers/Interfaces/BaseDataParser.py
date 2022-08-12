from typing import List
import abc

from ..Feature import Feature


class BaseDataParser(metaclass=abc.ABCMeta):
    def __init__(self, root, *args, **kwargs) -> None:
        self.root = root
        self._init_structure(*args, **kwargs)

    @abc.abstractmethod
    def _init_structure(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_feature(self, query: str) -> Feature:
        raise NotImplementedError

from typing import List, Any
import random


class Queue(object):
    """ Easy implementation of queue, use q.data to access all data. """

    data: list

    def __init__(self, max_size: int=100) -> None:
        self.max_size = max_size
        self.data = []

    def update(self, x):
        self.data.append(x)
        if len(self.data) > self.max_size:
            self.data.pop(0)


class DataPool(object):
    """
    A pooling data structure.
    If auto_resize is set to false, client has full control of when to call resize() to avoid calling it too much.
    """
    def __init__(self, max_size: int=100, auto_resize=True) -> None:
        self._max_size = max_size
        self._data = []
        self._auto_resize = auto_resize

    def append(self, data):
        self._data.append(data)
        if self._auto_resize:
            self.resize()

    def extend(self, data: List):
        self._data.extend(data)
        if self._auto_resize:
            self.resize()

    def sample(self, k=1):
        """
        Sample k elements from pool without replacement.
        """
        if self.empty():
            return None
        return random.sample(self._data, k)

    def choices(self, k=1):
        """
        Sample k elements from pool with replacement.
        """
        if self.empty():
            return None
        return random.choices(self._data, k)

    def clear(self):
        self._data = []

    def __len__(self):
        return len(self._data)
    
    def empty(self):
        return len(self._data) == 0

    def resize(self):
        if len(self._data) > self._max_size:
            self._data = random.sample(self._data, self._max_size)

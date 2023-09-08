from typing import List, Tuple


class TimeSequence(object):
    """
    Time sequece wrapper class of python objects.
    Consists of objects and intervals.
    """
    def __init__(self, objs, segments: List[Tuple[float, float]], *args, **kwargs) -> None:
        self.objs = objs
        self.segments = segments

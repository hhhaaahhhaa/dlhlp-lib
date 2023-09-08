from typing import List
import copy

from dlhlp_lib.interfaces import BaseCalculator, TimeSequence
from dlhlp_lib.utils.tool import segment2duration, expand


class FERCalculator(BaseCalculator):
    def __init__(self, fp: float):
        self._fp = fp
        self._state = {
            "matched": 0,
            "total": 0,
        }

    def _padding(seq: List, target_length: int) -> List:
        diff = len(seq) - target_length
        if diff >= 0:
            seq = seq[:target_length]
        else:
            padding = [seq[-1]] * (-diff)
            seq.extend(padding)
        return seq

    def exec(self, ref: TimeSequence, hyp: TimeSequence):
        ref_duration, hyp_duration = segment2duration(ref.segments, self._fp), segment2duration(hyp.segments, self._fp)
        ref_seq, hyp_seq = expand(ref.objs, ref_duration), expand(hyp.objs, hyp_duration)
        hyp_seq = self._padding(hyp_seq, target_length=len(ref_seq))

        for (x1, x2) in zip(ref_seq, hyp_seq):
            if x1 == x2:
                self._state["matched"] += 1
        self._state["total"] += len(ref_seq)

    def get_state(self) -> None:
        return copy.deepcopy(self._state)
    
    def clear(self) -> None:
        self._state = {
            "matched": 0,
            "total": 0,
        }

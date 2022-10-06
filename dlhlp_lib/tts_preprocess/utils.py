from tgt.core import IntervalTier


def get_alignment(tier_phone: IntervalTier):
    SILENCE = ["sil", "sp", "spn"]
    phones = []
    segments = []
    end_idx = 0
    for t in tier_phone._objects:
        s, e, p = t.start_time, t.end_time, t.text
        # Handle empty intervals
        if p == "":
            if s == 0:
                p = "sil"
            else:
                p = "sp"

        # Trim leading silences
        if phones == []:
            if p in SILENCE:
                continue
            else:
                pass
        phones.append(p)
        segments.append((s, e))
        if p not in SILENCE:
            end_idx = len(phones)

    segments = segments[:end_idx]
    phones = phones[:end_idx]

    return phones, segments


def representation_average(representation, durations, pad=0):
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            representation[i] = np.mean(
                representation[pos: pos + d], axis=0)
        else:
            representation[i] = pad
        pos += d
    return representation[: len(durations)]
 

class ImapWrapper(object):
    """
    Function object wrapper.
    """
    def __init__(self, func) -> None:
        self.f = func
    
    def __call__(self, task) -> bool:
        *args, ignore_errors = task
        try:
            self.f(*args)
        except:
            if ignore_errors:
                return False
            raise
        return True

import os
import glob
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class VocalSetInstance(object):
    """
    VocalSetInstance example:
        {'context': 'arpeggios',
        'id': 'female1-f1_arpeggios_breathy_a',
        'speaker': 'female1',
        'technique': 'breathy',
        'wav_path': '/mnt/d/Data/VocalSet/FULL/female1/arpeggios/breathy/f1_arpeggios_breathy_a.wav'}
    """

    id: str
    speaker: str
    wav_path: str
    context: str
    technique: str

    def __init__(self, id, speaker, wav_path, context, technique) -> None:
        self.id = id
        self.speaker = speaker
        self.wav_path = wav_path
        self.context = context
        self.technique = technique


class VocalSetRawParser(object):

    dataset: List[VocalSetInstance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/vocal_set.pkl"
        self._load()

    def _load(self):
        if not os.path.isfile(self._cache_path):
            self.parse()
        else:
            self.log("Loading cache...")
            with open(self._cache_path, 'rb') as f:
                data = pickle.load(f)
            self.dataset = list(data.values())

    def parse(self):
        self.log("Parsing...")
        data = {}

        for speaker in tqdm(os.listdir(f"{self.root}/FULL")):
            if "DS_Store" in speaker:
                continue
            for context in os.listdir(f"{self.root}/FULL/{speaker}"):
                if "DS_Store" in context:
                    continue
                for technique in os.listdir(f"{self.root}/FULL/{speaker}/{context}"):
                    if "DS_Store" in technique:
                        continue
                    for wav_path in glob.glob(f"{self.root}/FULL/{speaker}/{context}/{technique}/*.wav"):
                        id = f"{speaker}-{os.path.basename(wav_path)[:-4]}"
                        data[id] = VocalSetInstance(
                            id, speaker, wav_path, context, technique
                        )

        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[VocalSetRawParser]: ", msg)


if __name__ == "__main__":
    tmp = VocalSetRawParser("/mnt/d/Data/VocalSet")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

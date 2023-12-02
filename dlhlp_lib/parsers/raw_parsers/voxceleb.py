import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class Voxceleb1Instance(object):
    """
    Voxceleb1Instance example:
        {'id': 'id10003-BQxxhq4539A-00016',
        'speaker': 'id10003',
        'wav_path': '/mnt/d/Data/Voxceleb1-16kHz/dev/wav/id10003/BQxxhq4539A/00016.wav'}
    """

    id: str
    speaker: str
    wav_path: str

    def __init__(self, id, speaker, wav_path) -> None:
        self.id = id
        self.speaker = speaker
        self.wav_path = wav_path


class Voxceleb1RawParser(object):

    train: List[Voxceleb1Instance]
    dev: List[Voxceleb1Instance]
    test: List[Voxceleb1Instance]

    def __init__(self, root: str):
        self.root = root
        self.train = None
        self.dev = None
        self.test = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/voxceleb1.pkl"
        self._load()

    def _load(self):
        if not os.path.isfile(self._cache_path):
            self.parse()
        else:
            self.log("Loading cache...")
            with open(self._cache_path, 'rb') as f:
                data = pickle.load(f)
            for k in data:
                setattr(self, k.replace("-", "_"), list(data[k].values()))

    def parse(self):
        self.log("Parsing...")
        data = {
            "train": {},
            "dev": {},
            "test": {},
        }

        pathlib_root = Path(self.root)
        index2split = {"1": "train", "2": "dev", "3": "test"}
        with open(f"{self.root}/veri_test_class.txt", 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                if line == '\n':
                    continue
                [label, wav_id] = line.strip().split()
                speaker = wav_id.split('/')[0]
                id = wav_id.replace('/', '-')[:-4]
                x = list(pathlib_root.glob("*/wav/" + wav_id))
                wav_path = str(x[0])
                try:
                    assert os.path.isfile(wav_path)
                    data[index2split[label]][wav_id] = Voxceleb1Instance(
                        id, speaker, wav_path
                    )
                except:
                    print(f"Skip {id} due to missing file {wav_path}.")
               
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[Voxceleb1RawParser]: ", msg)


if __name__ == "__main__":
    tmp = Voxceleb1RawParser("/mnt/d/Data/Voxceleb1-16kHz")
    print(len(tmp.train))
    pprint.pprint(tmp.train[5].__dict__)

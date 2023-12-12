import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint
import pandas as pd
import ast


class UrbanSound8KInstance(object):
    """
    UrbanSound8KInstance example:
        {'class_id': 5,
        'class_name': 'engine_idling',
        'fold': 3,
        'id': '107228-5-0-0',
        'wav_path': '/mnt/d/Data/UrbanSound8K/audio/fold3/107228-5-0-0.wav'}
    """

    id: str
    fold: int
    class_id: int
    class_name: str
    wav_path: str

    def __init__(self, id, fold, class_id, class_name, wav_path) -> None:
        self.id = id
        self.fold = fold
        self.class_id = class_id
        self.class_name = class_name
        self.wav_path = wav_path


class UrbanSound8KRawParser(object):

    dataset: List[UrbanSound8KInstance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/urbansound8k.pkl"
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

        metadata = pd.read_csv(f'{self.root}/metadata/UrbanSound8K.csv')
        for i in tqdm(range(metadata.shape[0])):
            id = metadata.iloc[i, 0][:-4]
            fold = int(metadata.iloc[i, 5])
            class_id = int(metadata.iloc[i, 6])
            class_name = metadata.iloc[i, 7]

            wav_path = f"{self.root}/audio/fold{fold}/{id}.wav"
            try:
                assert os.path.isfile(wav_path)
                data[id] = UrbanSound8KInstance(
                    id, fold, class_id, class_name, wav_path
                )
            except:
                print(f"Skip {id} due to missing file {wav_path}.")
               
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[UrbanSound8KRawParser]: ", msg)


if __name__ == "__main__":
    tmp = UrbanSound8KRawParser("/mnt/d/Data/UrbanSound8K")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[500].__dict__)

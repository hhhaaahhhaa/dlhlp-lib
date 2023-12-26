import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class TAU2019UASInstance(object):
    """
    TAU2019UASInstance example:
        {'class_name': 'tram',
        'id': 'tram-lyon-1112-40515-a',
        'wav_path': '/mnt/d/Data/tau2019uas/TAU-urban-acoustic-scenes-2019-development/audio/tram-lyon-1112-40515-a.wav'}
    """

    id: str
    class_name: str
    wav_path: str

    def __init__(self, id, class_name, wav_path) -> None:
        self.id = id
        self.class_name = class_name
        self.wav_path = wav_path


class TAU2019UASRawParser(object):

    dataset: List[TAU2019UASInstance]

    def __init__(self, root: str):
        self.root = root
        self.train = None
        self.evaluate = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/tau2019uas.pkl"
        self._load()

    def _load(self):
        if not os.path.isfile(self._cache_path):
            self.parse()
        else:
            self.log("Loading cache...")
            with open(self._cache_path, 'rb') as f:
                data = pickle.load(f)
            for k in data:
                setattr(self, k, list(data[k].values()))

    def parse(self):
        self.log("Parsing...")
        data = {
            "train": {},
            "evaluate": {},
        }

        for split_name in data:
            with open(f'{self.root}/evaluation_setup/fold1_{split_name}.csv', "r") as f:
                for i, line in tqdm(enumerate(f)):
                    if i == 0:
                        continue
                    if line == '\n':
                        continue
                    filename, scene_label = line.strip().split('\t')
                    wav_path = f"{self.root}/{filename}"
                    id = filename[6:-4]
                    try:
                        assert os.path.isfile(wav_path)
                        data[split_name][id] = TAU2019UASInstance(
                            id, scene_label, wav_path
                        )
                    except:
                        print(f"Skip {id} due to missing file {wav_path}.")

        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[TAU2019UASRawParser]: ", msg)


if __name__ == "__main__":
    tmp = TAU2019UASRawParser("/mnt/d/Data/tau2019uas/TAU-urban-acoustic-scenes-2019-development")
    print(len(tmp.train))
    pprint.pprint(tmp.train[500].__dict__)

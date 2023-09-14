import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class CSS10Instance(object):
    """
    CSS10Instance example:
        {'id': 'achtgesichterambiwasse_0005',
        'text': 'und als ginge nur ihr Schatten mit der Dienerin und der '
                'Papierlaterne den Weg zu den Schatten.',
        'wav_path': '/mnt/d/Data/CSS10/german/achtgesichterambiwasse/achtgesichterambiwasse_0005.wav'}
    """

    id: str
    wav_path: str
    text: str

    def __init__(self, id, wav_path, text) -> None:
        self.id = id
        self.wav_path = wav_path
        self.text = text
    

class CSS10RawParser(object):

    dataset: List[CSS10Instance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None
        self.lang = str(self.root).split('/')[-1]

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache/css10", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/css10/{self.lang}.pkl"
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

        with open(f"{self.root}/transcript.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines)):
                if line == '\n':
                    continue
                wav_name, _, text, _ = line.strip().split('|')
                wav_path = f"{self.root}/{wav_name}"
                id = wav_name.split('/')[-1][:-4]
                try:
                    assert os.path.isfile(wav_path)
                    data[id] = CSS10Instance(
                        id, wav_path, text
                    )
                except:
                    print(f"Skip {id} due to missing file {wav_path}.")
        
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[CSS10RawParser]: ", msg)


if __name__ == "__main__":
    tmp = CSS10RawParser("/mnt/d/Data/CSS10/german")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

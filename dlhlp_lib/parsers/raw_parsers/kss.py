import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class KSSInstance(object):
    """
    KSSInstance example:
        {'en_text': 'Do you know much about cars?',
        'id': '1_0005.wav',
        'text': '차에 대해 잘 아세요?',
        'wav_path': '/mnt/d/Data/kss/1/1_0005.wav'}
    """

    id: str
    wav_path: str
    text: str
    en_text: str

    def __init__(self, id, wav_path, text, en_text) -> None:
        self.id = id
        self.wav_path = wav_path
        self.text = text
        self.en_text = en_text
    

class KSSRawParser(object):

    dataset: List[KSSInstance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/kss.pkl"
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

        with open(f"{self.root}/transcript.v.1.4.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines)):
                if line == '\n':
                    continue
                wav_name, _, text, _, _, en_text = line.strip().split("|")
                id = wav_name[2:-4]
                wav_path = f"{self.root}/{wav_name}"

                try:
                    assert os.path.isfile(wav_path)
                    data[id] = KSSInstance(
                        id, wav_path, text, en_text
                    )
                except:
                    print(f"Skip {id} due to missing file {wav_path}.")
        
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[KSSRawParser]: ", msg)


if __name__ == "__main__":
    tmp = KSSRawParser("/mnt/d/Data/kss")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

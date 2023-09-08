import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class LJSpeechInstance(object):
    """
    LJSpeechInstance example:
        {'id': 'LJ001-0006',
        'text': 'And it is worth mention in passing that, as an example of fine '
                'typography,',
        'wav_path': '/mnt/d/Data/LJSpeech-1.1/wavs/LJ001-0006.wav'}
    """

    id: str
    wav_path: str
    text: str

    def __init__(self, id, wav_path, text) -> None:
        self.id = id
        self.wav_path = wav_path
        self.text = text


class LJSpeechRawParser(object):

    dataset: List[LJSpeechInstance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/ljspeech.pkl"
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

        with open(f"{self.root}/metadata.csv", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines)):
                if line == '\n':
                    continue
                id, _, text = line.strip().split("|")
                if text[-1].isalpha():  # add missing periods
                    text += '.'
                wav_path = f"{self.root}/wavs/{id}.wav"

                try:
                    assert os.path.isfile(wav_path)
                    data[id] = LJSpeechInstance(
                        id, wav_path, text
                    )
                except:
                    print(f"Skip {id} due to missing file {wav_path}.")

        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[LJSpeechRawParser]: ", msg)


if __name__ == "__main__":
    tmp = LJSpeechRawParser("/mnt/d/Data/LJSpeech-1.1")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class JSUTInstance(object):
    """
    JSUTInstance example:
        {'id': 'BASIC5000_0006',
        'text': '週に四回、フランスの授業があります。',
        'wav_path': '/mnt/d/Data/jsut_ver1.1/basic5000/wav/BASIC5000_0006.wav'}
    """

    id: str
    wav_path: str
    text: str

    def __init__(self, id, wav_path, text) -> None:
        self.id = id
        self.wav_path = wav_path
        self.text = text


class JSUTRawParser(object):

    basic5000: List[JSUTInstance]
    countersuffix26: List[JSUTInstance]
    loanword128: List[JSUTInstance]
    onomatopee300: List[JSUTInstance]
    precedent130: List[JSUTInstance]
    repeat500: List[JSUTInstance]
    travel1000: List[JSUTInstance]
    utparaphrase512: List[JSUTInstance]
    voiceactress100: List[JSUTInstance]

    def __init__(self, root: str):
        self.root = root
        self.basic5000 = None
        self.countersuffix26 = None
        self.loanword128 = None
        self.onomatopee300 = None
        self.precedent130 = None
        self.repeat500 = None
        self.travel1000 = None
        self.utparaphrase512 = None
        self.voiceactress100 = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/jsut.pkl"
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
        data = {}

        # dsets
        dsets = [
            "basic5000",
            "countersuffix26",
            "loanword128",
            "onomatopee300",
            "precedent130",
            "repeat500",
            "travel1000",
            "utparaphrase512",
            "voiceactress100"
        ]
        for dset in dsets:
            data[dset] = {}
            with open(f"{self.root}/{dset}/transcript_utf8.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, total=len(lines)):
                    if line == '\n':
                        continue
                    [id, text] = line.strip().split(":")
                    wav_path = f"{self.root}/{dset}/wav/{id}.wav"

                    try:
                        assert os.path.isfile(wav_path)
                        data[dset][id] = JSUTInstance(
                            id, wav_path, text
                        )
                    except:
                        print(f"Skip {id} due to missing file {wav_path}.")

        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[JSUTRawParser]: ", msg)


if __name__ == "__main__":
    tmp = JSUTRawParser("/mnt/d/Data/jsut_ver1.1")
    print(len(tmp.basic5000))
    pprint.pprint(tmp.basic5000[5].__dict__)

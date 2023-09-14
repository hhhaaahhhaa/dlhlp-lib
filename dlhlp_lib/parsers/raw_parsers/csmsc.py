import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class CSMSCInstance(object):
    """
    CSMSCInstance example:
        
    """

    id: str
    wav_path: str
    text: str

    def __init__(self, id, wav_path, text) -> None:
        self.id = id
        self.wav_path = wav_path
        self.text = text
    

class CSMSCRawParser(object):

    dataset: List[CSMSCInstance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/csmsc.pkl"
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

        with open(f"{self.root}/ProsodyLabeling/000001-010000.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines)):
                if line == '\n':
                    continue
                wav_name, text = line.strip().split("\t")
                parsed_text = ""
                st = 0
                while st < len(text):
                    if text[st] == "#":
                        st += 2
                    else:
                        parsed_text += text[st]
                        st += 1
                
                wav_path = f"{self.root}/Wave/{wav_name}.wav"
                id = wav_name
                try:
                    assert os.path.isfile(wav_path)
                    data[id] = CSMSCInstance(
                        id, wav_path, text
                    )
                except:
                    print(f"Skip {id} due to missing file {wav_path}.")
        
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[CSMSCRawParser]: ", msg)


if __name__ == "__main__":
    tmp = CSMSCRawParser("/mnt/d/Data/csmsc")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

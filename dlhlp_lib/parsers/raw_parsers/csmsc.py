import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class CSMSCInstance(object):
    """
    CSMSCInstance example:
        {'id': '000006',
        'text': '身长#2约#1五尺#1二寸#1五分#2或#1以上#4。',
        'textgrid_path': '/mnt/d/Data/CSMSC/PhoneLabeling/000006.interval',
        'wav_path': '/mnt/d/Data/CSMSC/Wave/000006.wav'}
    """

    id: str
    wav_path: str
    textgrid_path: str
    text: str
    prosody_text: str

    def __init__(self, id, wav_path, textgrid_path, text, prosody_text) -> None:
        self.id = id
        self.wav_path = wav_path
        self.textgrid_path = textgrid_path
        self.text = text
        self.prosody_text = prosody_text
    

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
            for i, line in enumerate(tqdm(lines, total=len(lines))):
                if line == '\n':
                    continue
                if i % 2 == 1:
                    continue
                wav_name, prosody_text = line.strip().split("\t")
                parsed_text = ""
                st = 0
                while st < len(prosody_text):
                    if prosody_text[st] == "#":
                        st += 2
                    else:
                        parsed_text += prosody_text[st]
                        st += 1

                textgrid_path = f"{self.root}/PhoneLabeling/{wav_name}.interval"
                wav_path = f"{self.root}/Wave/{wav_name}.wav"
                id = wav_name
                try:
                    assert os.path.isfile(wav_path)
                    data[id] = CSMSCInstance(
                        id, wav_path, textgrid_path, parsed_text, prosody_text
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
    tmp = CSMSCRawParser("/mnt/d/Data/CSMSC")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

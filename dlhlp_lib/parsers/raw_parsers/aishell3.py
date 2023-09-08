import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class AISHELL3Instance(object):
    """
    AISHELL3Instance example:
        {'id': 'SSB00050006',
        'labeled_phns': 'wu2 bai3 % si4 shi2 wu3 wan4 % qi1 qian1 % yi1 bai3 % er4 '
                        'shi2 er4 $',
        'labeled_text': '五百%四十五万%七千%一百%二十二$',
        'phns': 'wu2 bai3 si4 shi2 wu3 wan4 qi1 qian1 yi1 bai3 er4 shi2 er4',
        'speaker': 'SSB0005',
        'text': '五百四十五万七千一百二十二',
        'wav_path': '/mnt/d/Data/AISHELL-3/train/wav/SSB0005/SSB00050006.wav'}
    """

    id: str
    speaker: str
    wav_path: str
    text: str
    phns: str
    labeled_text: Optional[str]
    labeled_phns: Optional[str]

    def __init__(self, id, speaker, wav_path, text, phns, labeled_text, labeled_phns) -> None:
        self.id = id
        self.speaker = speaker
        self.wav_path = wav_path
        self.text = text
        self.phns = phns
        self.labeled_text = labeled_text
        self.labeled_phns = labeled_phns


class AISHELL3RawParser(object):

    train_set: List[AISHELL3Instance]
    test_set: List[AISHELL3Instance]

    def __init__(self, root: str):
        self.root = root
        self.train_set = None
        self.test_set = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/aishell3.pkl"
        self._load()

    def _load(self):
        if not os.path.isfile(self._cache_path):
            self.parse()
        else:
            self.log("Loading cache...")
            with open(self._cache_path, 'rb') as f:
                data = pickle.load(f)
            self.train_set = list(data["train"].values())
            self.test_set = list(data["test"].values())
    
    def parse(self):
        self.log("Parsing...")
        data = {}

        # train/test content
        for split in ["train", "test"]:
            data[split] = {}
            with open(f"{self.root}/{split}/content.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, total=len(lines)):
                    if line == '\n':
                        continue
                    [wav_name, transcript] = line.strip().split("\t")
                    wav_name = wav_name[:-4]
                    id = wav_name
                    speaker = wav_name[:-4]
                    wav_path = f"{self.root}/{split}/wav/{speaker}/{wav_name}.wav"

                    # split text
                    tokens = transcript.split()
                    text, phns = [], []
                    for j, x in enumerate(tokens):
                        if j % 2 == 0:
                            text.append(x)
                        else:
                            phns.append(x)
                    text = "".join(text)
                    phns = " ".join(phns)

                    try:
                        assert os.path.isfile(wav_path)
                        data[split][id] = AISHELL3Instance(
                            id, speaker, wav_path, text, phns, None, None
                        )
                    except:
                        print(f"Skip {id} due to missing file {wav_path}.")

        # train labels
        with open(f"{self.root}/train/label_train-set.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in tqdm(enumerate(lines), total=len(lines)):
                if i < 5 or line == '\n':
                    continue
                wav_name, phns, text = line.strip().split('|')
                data["train"][wav_name].labeled_text = text
                data["train"][wav_name].labeled_phns = phns

        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[AISHELL3RawParser]: ", msg)


if __name__ == "__main__":
    tmp = AISHELL3RawParser("/mnt/d/Data/AISHELL-3")
    print(len(tmp.train_set), len(tmp.test_set))
    pprint.pprint(tmp.train_set[5].__dict__)

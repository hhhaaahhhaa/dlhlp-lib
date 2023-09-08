import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class LibriTTSInstance(object):
    """
    LibriTTSInstance example:
        {'chapter': '1241',
        'id': '103_1241_000014_000003',
        'speaker': '103',
        'text': 'Her face was small, white and thin, also much freckled; her mouth '
                'was large and so were her eyes, which looked green in some lights '
                'and moods and gray in others.',
        'wav_path': '/mnt/d/Data/LibriTTS/train-clean-100/103/1241/103_1241_000014_000003.wav'}
    """

    id: str
    speaker: str
    wav_path: str
    text: str
    chapter: str

    def __init__(self, id, speaker, wav_path, text, chapter) -> None:
        self.id = id
        self.speaker = speaker
        self.wav_path = wav_path
        self.text = text
        self.chapter = chapter


class LibriTTSRawParser(object):

    train_clean_100: List[LibriTTSInstance]
    train_clean_360: List[LibriTTSInstance]
    train_other_500: List[LibriTTSInstance]
    dev_clean: List[LibriTTSInstance]
    dev_other: List[LibriTTSInstance]
    test_clean: List[LibriTTSInstance]
    test_other: List[LibriTTSInstance]

    def __init__(self, root: str):
        self.root = root
        self.train_clean_100 = None
        self.train_clean_360 = None
        self.train_other_500 = None
        self.dev_clean = None
        self.dev_other = None
        self.test_clean = None
        self.test_other = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/libritts.pkl"
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
        data = {}

        # dsets
        dsets = [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]
        for dset in dsets:
            if not os.path.isdir(f"{self.root}/{dset}"):
                continue
            data[dset] = {}
            for speaker in tqdm(os.listdir(f"{self.root}/{dset}"), desc=dset):
                for chapter in os.listdir(f"{self.root}/{dset}/{speaker}"):
                    for filename in os.listdir(f"{self.root}/{dset}/{speaker}/{chapter}"):
                        if filename[-4:] != ".wav":
                            continue
                        id = filename[:-4]
                        wav_path = f"{self.root}/{dset}/{speaker}/{chapter}/{id}.wav"
                        text_path = f"{self.root}/{dset}/{speaker}/{chapter}/{id}.normalized.txt"
                        with open(text_path, "r", encoding="utf-8") as f:
                            text = f.readline().strip("\n")
                        
                        try:
                            assert os.path.isfile(wav_path)
                            data[dset][id] = LibriTTSInstance(
                                id, speaker, wav_path, text, chapter
                            )
                        except:
                            print(f"Skip {id} due to missing file {wav_path}.")

        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[LibriTTSRawParser]: ", msg)


if __name__ == "__main__":
    tmp = LibriTTSRawParser("/mnt/d/Data/LibriTTS")
    print(len(tmp.train_clean_100))
    pprint.pprint(tmp.train_clean_100[5].__dict__)

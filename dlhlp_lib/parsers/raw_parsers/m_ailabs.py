import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint


class MAILABSInstance(object):
    """
    MAILABSInstance example:
        {'book': 'le_pays_des_fourrures',
        'id': 'le_pays_des_fourrures_1_10_f000006',
        'speaker': 'bernard',
        'text': 'Quant aux courageux et dévoués Esquimaux, après avoir reçu '
                'flegmatiquement les affectueux remerciements du lieutenant et de sa '
                "compagne, ils n'avaient même pas voulu venir au fort.",
        'wav_path': '/work/u7663915/Data/M-AILABS/fr_FR/male/bernard/le_pays_des_fourrures/wavs/le_pays_des_fourrures_1_10_f000006.wav'}
    """

    id: str
    speaker: str
    wav_path: str
    text: str
    book: str

    def __init__(self, id, speaker, wav_path, text, book) -> None:
        self.id = id
        self.speaker = speaker
        self.wav_path = wav_path
        self.text = text
        self.book = book
    

class MAILABSRawParser(object):

    dataset: List[MAILABSInstance]

    def __init__(self, root: str):
        self.root = root
        self.dataset = None
        self.lang = self.root.split('/')[-1]

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache/m_ailabs", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/m_ailabs/{self.lang}.pkl"
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

        if self.lang == "fr_FR":  # french has wrong format, might be fixed in future by official
            paths = [f"{self.root}/male", f"{self.root}/female"]
        else:
            paths = [f"{self.root}/by_book/male", f"{self.root}/by_book/female"]
        for path in paths:
            if not os.path.isdir(path):
                continue
            speakers = os.listdir(path)
            for speaker in tqdm(speakers):
                if self.lang == "fr_FR" and speaker == "nadine_eckert_boulet":  # french is partially corrupted, might be fixed in future by official
                    continue
                books = os.listdir(f"{path}/{speaker}")
                for book in books:
                    if not os.path.isdir(f"{path}/{speaker}/{book}"):
                        continue
                    dirpath = f"{path}/{speaker}/{book}"
                    with open(f"{dirpath}/metadata.csv", 'r', encoding='utf-8') as f:
                        for line in f:
                            if line == "\n":
                                continue
                            wav_name, origin_text, text = line.strip().split("|")
                            wav_path = f"{dirpath}/wavs/{wav_name}.wav"
                            id = wav_name
                            try:
                                assert os.path.isfile(wav_path)
                                data[id] = MAILABSInstance(
                                    id, speaker, wav_path, text, book
                                )
                            except:
                                print(f"Skip {id} due to missing file {wav_path}.")
        
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[MAILABSRawParser]: ", msg)


if __name__ == "__main__":
    tmp = MAILABSRawParser("/work/u7663915/Data/M-AILABS/ru_RU")
    print(len(tmp.dataset))
    pprint.pprint(tmp.dataset[5].__dict__)

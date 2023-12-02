import os
from tqdm import tqdm
from typing import List, Optional
import pickle
import pprint
import pandas as pd
import ast


class FMAMusicInstance(object):
    """
    FMAMusicInstance example:
        {'genre': 'Electronic',
        'id': '13668',
        'wav_path': '/mnt/d/Data/fma/fma_small/013/013668.mp3'}
    """

    id: str
    genre: str
    wav_path: str

    def __init__(self, id, genre, wav_path) -> None:
        self.id = id
        self.genre = genre
        self.wav_path = wav_path


class FMAMusicRawParser(object):

    training: List[FMAMusicInstance]
    validation: List[FMAMusicInstance]
    test: List[FMAMusicInstance]
    genres: List[str]

    def __init__(self, root: str):
        self.root = root
        self.training = None
        self.validation = None
        self.test = None

        _current_dir = os.path.dirname(__file__)
        os.makedirs(f"{_current_dir}/_cache", exist_ok=True)
        self._cache_path = f"{_current_dir}/_cache/fma_music.pkl"
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
            "training": {},
            "validation": {},
            "test": {},
        }

        tracks = load(f'{self.root}/fma_metadata/tracks.csv')

        # Currently only support fma_small
        small = tracks[tracks['set', 'subset'] <= 'small']
        # print(small.shape)

        for split_name in ['training', 'validation', 'test']:
            ds = small.loc[tracks['set', 'split'] == split_name]
            y = ds.loc[:, ('track', 'genre_top')]
            track_ids = list(y.index)
            for i, id in tqdm(enumerate(track_ids)):
                tid_str = '{:06d}'.format(id)
                wav_path = f"{self.root}/fma_small/{tid_str[:3]}/{tid_str}.mp3"
                genre = y.iloc[i]
                if id in [99134, 108925, 133297]:  # skip these corrupted ids, this is an official issue
                    continue
                try:
                    assert os.path.isfile(wav_path)
                    data[split_name][str(id)] = FMAMusicInstance(
                        id=str(id), genre=genre, wav_path=wav_path
                    )
                except:
                    print(f"Skip {id} due to missing file {wav_path}.")
               
        self.log("Generating cache...")
        with open(self._cache_path, 'wb') as f:
            pickle.dump(data, f)
        self._load()

    def log(self, msg):
        print(f"[FMAMusicRawParser]: ", msg)


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


if __name__ == "__main__":
    tmp = FMAMusicRawParser("/mnt/d/Data/fma")
    print(len(tmp.training))
    pprint.pprint(tmp.training[500].__dict__)

import os
from typing import List
import pickle
from tqdm import tqdm

from .Interfaces.BaseIOObject import BaseIOObject
from .Interfaces.BaseQueryParser import BaseQueryParser


class Feature(object):
    """
    Template class for single feature.
    """
    def __init__(self, name: str, root: str, parser: BaseQueryParser, io: BaseIOObject, enable_cache=False):
        self.name = name
        self.root = root
        self.query_parser = parser
        self.io = io
        self._data = None
        self._enable_cache = enable_cache

    def read_all(self, refresh=False):
        if self._data is not None:  # cache already loaded
            pass
        if not self._enable_cache:
            self.log("Cache not supported...")
            raise NotImplementedError
        cache_path = self.query_parser.get_cache()
        if not os.path.isfile(cache_path) or refresh:
            self.log("Generating cache...")
            data = {}
            filenames = self.query_parser.get_all(extension=self.io.extension)
            for filename in tqdm(filenames):
                data[filename] = self.read_from_filename(filename)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self._data = data
        else:
            self.log("Loading cache...")
            with open(cache_path, 'rb') as f:
                self._data = pickle.load(f)

    def read_filename(self, query) -> str:
        filenames = self.query_parser.get(query)
        assert len(filenames) == 1
        return filenames[0]

    def read_filenames(self, query) -> List[str]:
        return self.query_parser.get(query)
    
    def read_from_query(self, query):
        filename = self.read_filename(query)
        return self.read_from_filename(filename)

    def read_from_filename(self, filename):
        if self._data is not None:
            return self._data[filename]
        return self.io.readfile(f"{self.root}/{self.name}/{filename}{self.io.extension}")
    
    def save(self, input, query):
        filenames = self.query_parser.get(query)
        assert len(filenames) == 1
        path = f"{self.root}/{self.name}/{filenames[0]}{self.io.extension}"
        self.io.savefile(input, path)

    def log(self, msg):
        print(f"[Feature ({self.root}/{self.name})]: ", msg)

import _pickle as pickle
import os.path
import random


class PickleCache(object):
    def __init__(self, pickle_suffix='.pickle'):
        self.pickle_suffix = pickle_suffix

    def pickle_name(self, fpath):
        return fpath + self.pickle_suffix

    def exists(self, fpath):
        return os.path.isfile(self.pickle_name(fpath))

    def write(self, fpath, val):
        pickle_file_path = self.pickle_name(fpath)
        pf = open(pickle_file_path, 'wb')
        pickle.dump(val, pf)
        return val

    def read(self, fpath, clear_cache=False):
        if (not self.exists(fpath)) or clear_cache:
            val = self.load_object(fpath)
            return self.write(fpath, val)
        else:
            pickle_file = open(self.pickle_name(fpath), 'rb')
            return pickle.load(pickle_file)

    def load_object(self, fpath):
        raise NotImplemented()


def random_json_file_subset(infile, outfile, size):
    with open(infile, 'rb') as source:
        data = [(random.random(), line) for line in source]
    with open(outfile, 'wb') as dest:
        for r, line in data[:size]:
            dest.write(line)

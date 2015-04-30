import os
import gzip
import cPickle as pickle
from glob import glob
import helpers.classes as ch

def load_single_file(filename):
    with gzip.GzipFile(filename, 'rb') as f:
            content = pickle.load(f)

    basename = os.path.basename(filename)
    label = ''.join([i if ord(i) < 128 else ' ' for i in basename])
    cls, cls_label = ch.get_class(basename)
    cls_short = ch.get_class_short(cls)

    content['label'] = label
    content['filename'] = basename
    content['class'] = cls
    content['class_label'] = cls_label
    content['class_short'] = cls_short

    return content

def load_files(directory, filter_unfinished=True):
    filenames = glob(directory + '*.pkl')
    loaded = map(load_single_file, filenames)
    if filter_unfinished:
        loaded = filter(lambda x: 'done' in x and x['done'], loaded)
    return loaded
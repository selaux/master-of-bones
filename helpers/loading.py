import os
import gzip
import cPickle as pickle
from glob import glob
import helpers.classes as ch

def load_single_file(filename):
    with gzip.GzipFile(filename, 'rb') as f:
            content = pickle.load(f)

    content.update(get_info_from_filename(filename))

    return content

def get_info_from_filename(filename):
    basename = os.path.basename(filename)
    label = ''.join([i if ord(i) < 128 else ' ' for i in basename])
    cls, cls_label = ch.get_class(basename)
    cls_short = ch.get_class_short(cls)
    color = ch.get_classed_color(cls, label=basename)

    return {
        'label': label,
        'filename': basename,
        'class': cls,
        'class_label': cls_label,
        'class_short': cls_short,
        'color': color
    }

def load_files(directory, filter_unfinished=True):
    filenames = glob(os.path.join(directory, '*.pkl'))
    loaded = map(load_single_file, filenames)
    print('Loaded {} Files'.format(len(loaded)))
    if filter_unfinished:
        loaded = filter(lambda x: 'done' in x and x['done'], loaded)
    print('Found {} Files wich have the done flag set'.format(len(loaded)))
    return loaded

def save_files(directory, outlines):
    for outline in outlines:
        save_single_file(os.path.join(directory, outline['filename']), outline)

def save_single_file(path, outline):
    with gzip.open(path, 'wb') as f:
        pickle.dump(outline, f)

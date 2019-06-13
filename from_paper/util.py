import os
from PIL import Image
import numpy as np


def read_images(path, sz=None):
    c = 0
    x, y = [], []
    for dir_name, dir_names, filenames in os.walk(path):
        for sub_dir_name in dir_names:
            subject_path = os.path.join(dir_name, sub_dir_name)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given )
                    if sz is not None:
                        im = im.resize(sz, Image.ANTIALIAS)
                    x. append(np.asarray(im, dtype=np.uint8))
                    y. append(c)
                except IOError as e_io:
                    print("I/O error {0}". format(e_io))
                except Exception as e:
                    print(" Unexpected error :", e)
                    raise
        c = c + 1
    return [x, y]


def as_row_matrix(x):
    if len(x) == 0:
        return np.array([])
    mat = np.empty((0, x[0]. size), dtype=x[0]. dtype)
    for row in x:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def as_column_matrix(x):
    if len(x) == 0:
        return np.array([])
    mat = np.empty((x[0].size, 0), dtype=x[0].dtype)
    for col in x:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


def normalize(x, low, high, dtype=None):
    x = np.asarray(x)
    minx, maxx = np.min(x), np.max(x)
    # normalize to [0...1].
    x = x - float(minx)
    x = x / float((maxx - minx))
    # scale to [ low ... high ].
    x = x * (high - low)
    x = x + low
    if dtype is None:
        return np.asarray(x)
    return np.asarray(x, dtype=dtype)


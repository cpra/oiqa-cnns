
'''
Randomly sample patch locations from training, validation, and test images.
~ Christopher Pramerdorfer
'''

import cv2
import numpy as np

import os
import sys
import json

seed = 123
np.random.seed(seed)

# -----

source = '/mnt/storage/datasets/image-quality-analysis/LIVE2/original/refimgs'  # directory to read from
tvt = '/home/chris/projects/caip-paper/datasets/LIVE2/regression/tvt.json'  # train-val-test split info
dest = '/home/chris/projects/caip-paper/datasets/LIVE2/regression/patchlocs-40-32-32' # directory to save to

cfg = {
    'train': {
        'num': 200,
        'sz': 40
    },
    'val': {
        'num': 200,
        'sz': 32
    },
    'test': {
        'num': 200,
        'sz': 32
    }
}

# -----

assert(os.path.isdir(source))
assert(not os.path.exists(dest))

os.mkdir(dest)

# -----

print('Loading split information ...')

with open(tvt, 'r') as f:
    tvt = json.load(f)

    for s in tvt:
        print(' {} {} images'.format(len(tvt[s]), s))

# -----

print('Locating images ...')

ims = [f for f in os.listdir(source)]
print(' {} images found'.format(len(ims)))

assert(len(ims) == sum([len(tvt[s]) for s in tvt]))

for s in tvt:
    num = cfg[s]['num']
    sz = cfg[s]['sz']

    print('Split {}, {} patches of size {}'.format(s, num, sz))

    for imf in tvt[s]:
        imp = os.path.join(source, imf)
        assert(os.path.isfile(imp))

        im = cv2.imread(imp, cv2.IMREAD_UNCHANGED)
        mask  = np.zeros(im.shape[:2], dtype=np.uint16)
        h, w = im.shape[:2]

        print(' {} ({}x{})'.format(os.path.basename(imp), w, h))

        with open(os.path.join(dest, '{}.txt'.format(os.path.splitext(os.path.basename(imp))[0])), 'w') as f:

            for n in range(num):
                x0 = np.random.randint(0, w - sz)
                y0 = np.random.randint(0, h - sz)
                x1 = x0 + sz
                y1 = y0 + sz

                f.write('{} {} {} {}\n'.format(x0, x1, y0, y1))  # x1, y1 are exclusive

                mask[y0:y1, x0:x1] += 1

        vmask = mask * 32
        vmask[vmask > 255] = 255

        cv2.imshow('mask', vmask / 255)
        cv2.waitKey(250)

print('Saved results to "{}"'.format(dest))
print(' Format is x0 x1 y0 x1 (x1 and y1 are exclusive)')

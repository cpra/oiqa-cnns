
'''
Resize the RAISE dataset (or a subset) and save as PNG.
~ Christopher Pramerdorfer
'''

import cv2

import os
import sys
import random

source = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE'  # directory to read from
dest = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images'  # directory to save to
scale = 0.20  # scale factor
interpolation = cv2.INTER_AREA  # interpolation mode (AREA is best for decimation)

num_save = 1000 + 200 + 100  # number of images to choose randomly for saving (negative = all)
save_rng = 1337  # rng seed for selecting which images to save

# -------

if not os.path.isdir(source):
    sys.exit('"{}" is no directory'.format(source))

if os.path.exists(dest):
    sys.exit('"{}" already exists'.format(dest))

os.mkdir(dest)

# -------

ims = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith('.TIF')])

print('{} TIF images found'.format(len(ims)))
assert(len(ims) == 8156)

# -------

if num_save >= 0:
    print('Selecting subset of {} images'.format(num_save))
    assert(num_save <= len(ims))

    random.seed(save_rng)

    idx = list(range(len(ims)))
    random.shuffle(idx)
    idx = idx[:num_save]

    ims = [ims[i] for i in idx]

# -------

for imp in ims:
    im = cv2.imread(imp, cv2.IMREAD_UNCHANGED)
    imr = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
    cv2.imwrite(os.path.join(dest, '{}.png'.format(os.path.splitext(os.path.basename(imp))[0])), imr)

    print(' {} : {}x{} -> {}x{}'.format(os.path.basename(imp), im.shape[1], im.shape[0], imr.shape[1], imr.shape[0]))


'''
Randomly assign images in a folder to a training, validation, and test set.
~ Christopher Pramerdorfer
'''

import os
import sys
import pickle
import random
import json

rng = 1337  # rng seed
frac_test = 0.1535  # test fraction
frac_val = 0.0905  # val fraction of training set (after removing test samples)

source = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images'  # directory to read from
dest = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/tvt-{}-{}-{}.json'.format(rng, frac_test, frac_val)  # file to save to
imext = '.png'  # file ending of images

# -------

if not os.path.isdir(source):
    sys.exit('"{}" is no directory'.format(source))

if os.path.exists(dest):
    sys.exit('"{}" already exists'.format(dest))

# -------

ims = sorted([f for f in os.listdir(source) if f.endswith(imext)])
print('{} images found'.format(len(ims)))

random.seed(rng)
random.shuffle(ims)

idx = int(len(ims)*(1.0-frac_test))

itest = ims[idx:]
ims = ims[:idx]

idx = int(len(ims)*(1.0-frac_val))

ival = ims[idx:]
itrain = ims[:idx]

print(' {} training images'.format(len(itrain)))
for im in itrain:
    print('  {}'.format(im))

print(' {} validation images'.format(len(ival)))
for im in ival:
    print('  {}'.format(im))

print(' {} test images'.format(len(itest)))
for im in itest:
    print('  {}'.format(im))

with open(dest, 'w') as f:
    json.dump({'train': itrain, 'val': ival, 'test': itest}, f, indent=2, separators=(',', ': '))

print('Split information saved to "{}"'.format(dest))

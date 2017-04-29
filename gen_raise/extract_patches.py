
'''
Extract patches for multi-label classification.
~ Christopher Pramerdorfer
'''

import cv2
import numpy as np
import h5py

import os
import sys
import json

# ------

patch_locations = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/patchlocs'
split_file = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/tvt.json'
dest = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/patches'

# ------

tasks = {
    'jpeg' : {
        0 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg/q20', '.jpg', 'q20'],
        1 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg/q35', '.jpg', 'q35'],
        2 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg/q50', '.jpg', 'q50'],
        3 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg/q65', '.jpg', 'q65'],
        4 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg/q80', '.jpg', 'q80'],
        5 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg/q95', '.jpg', 'q95'],
        6 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images', '.png', 'o'],
    },
    'jp2' : {
        0 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2/r32', '.png', 'r32'],
        1 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2/r24', '.png', 'r24'],
        2 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2/r20', '.png', 'r20'],
        3 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2/r16', '.png', 'r16'],
        4 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2/r12', '.png', 'r12'],
        5 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2/r8', '.png', 'r8'],
        6 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images', '.png', 'o'],
    },
    'gblur' : {
        0 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur/s2.00', '.png', 's2.00'],
        1 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur/s1.50', '.png', 's1.50'],
        2 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur/s1.25', '.png', 's1.25'],
        3 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur/s1.00', '.png', 's1.00'],
        4 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur/s0.75', '.png', 's0.75'],
        5 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur/s0.50', '.png', 's0.50'],
        6 : ['/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images', '.png', 'o'],
    }
}

# -------

assert(os.path.isdir(patch_locations))
assert(os.path.isfile(split_file))
assert(not os.path.exists(dest))

os.makedirs(dest)

# -------

print('Loading split information ...')

with open(split_file, 'r') as f:
    splits = json.load(f)

    print(' Split file contains {} splits:'.format(len(splits)))
    for s in splits:
        print('  - {} ({} images)'.format(s, len(splits[s])))

# -------

print('Loading patch locations ...')

patchloc = {}

for s in splits:
    print(' Split {}'.format(s))

    ims = splits[s]

    # -------

    for i, imfile in enumerate(ims):
        key = os.path.splitext(imfile)[0]
        assert(key not in patchloc)

        fn = os.path.join(patch_locations, '{}.txt'.format(key))
        patchloc[key] = np.loadtxt(fn).astype(np.int32)

# -------

print('Extracting patches ...')

for task in tasks:
    print(' Task {}'.format(task))

    save_dir = os.path.join(dest, task)
    os.mkdir(save_dir)

    task_info = tasks[task]

    for s in splits:
        ims = splits[s]

        # allocate memory

        num_samples = 0

        for im in ims:
            key = os.path.splitext(im)[0]
            num_samples += patchloc[key].shape[0] * len(task_info)

        loc = patchloc[key][0, :]
        dx = loc[1] - loc[0]
        dy = loc[3] - loc[2]

        print('  Split {}: {} patches of size {}x{}'.format(s, num_samples, dx, dy))

        X = np.empty((num_samples, 3, dy, dx), dtype=np.uint8)
        y = np.empty((num_samples,), dtype=np.uint8)

        print('   Data size: {:.2f}GB, Label size: {:.2f}MB'.format(1.0 * X.nbytes / 2**30, 1.0 * y.nbytes / 2**20))

        # write task class file

        with open(os.path.join(save_dir, 'classes.txt'), 'w') as f:
            for cid in task_info:
                f.write('{} {}\n'.format(cid, task_info[cid][2]))

        # extract patches

        sid = 0

        with open(os.path.join(save_dir, 'image_sources_{}.txt'.format(s)), 'w') as f_sources:
            with open(os.path.join(save_dir, 'image_classes_{}.txt'.format(s)), 'w') as f_classes:
                for cid in task_info:
                    imdir = task_info[cid][0]
                    imext = task_info[cid][1]

                    print('   Path "{}", class {}'.format(imdir, cid))

                    for imx in ims:
                        key = os.path.splitext(imx)[0]

                        image_path = os.path.join(imdir, '{}{}'.format(key, imext))
                        assert(os.path.isfile(image_path))

                        f_classes.write('{} {}\n'.format(image_path, cid))

                        im = cv2.imread(image_path, cv2.IMREAD_COLOR)

                        for x0, x1, y0, y1 in patchloc[key]:
                            f_sources.write('{}\n'.format(image_path))

                            X[sid, :, :, :] = np.rollaxis(im[y0:y1, x0:x1, :], 2)
                            y[sid] = cid

                            sid += 1

        assert(sid == num_samples)

        # write dataset

        print('   Writing to disk ...')

        h5f = h5py.File(os.path.join(save_dir, '{}.h5'.format(s)), 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('y', data=y)
        h5f.close()


'''
Extract images affected with JP2K compression artefacts.
~ Christopher Pramerdorfer
'''

import cv2

import os
import subprocess

src = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images'
src_imext = '.png'

dst_root = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jp2'
ratios = (8, 12, 16, 20, 24, 32)

# -------

assert(os.path.isdir(src))
assert(not os.path.exists(dst_root))

# -------

os.mkdir(dst_root)

for r in ratios:
    os.mkdir(os.path.join(dst_root, 'r{}'.format(r)))

# -------

ims = sorted([os.path.join(src, f) for f in os.listdir(src) if f.endswith(src_imext)])
print('{} images found'.format(len(ims)))

for imp in ims:
    print(os.path.basename(imp))

    im = cv2.imread(imp, cv2.IMREAD_UNCHANGED)
    imf = os.path.splitext(os.path.basename(imp))[0]

    for r in ratios:
        dest = os.path.join(dst_root, 'r{}'.format(r), '{}.png'.format(imf))

        # conver to jp2

        fnull = open(os.devnull, 'w')

        subprocess.check_call([
            "image_to_j2k",
            '-i', imp,
            '-r', str(r),
            '-o', '/tmp/im_temp.jp2'
        ], stdout=fnull, stderr=subprocess.STDOUT)

        print(' r={}, file size {:.1f}K'.format(r, os.path.getsize('/tmp/im_temp.jp2') / 1024))

        # and then as PNG to avoid problems while reading the files

        subprocess.check_call([
            "convert",
            "/tmp/im_temp.jp2",
            dest
        ])

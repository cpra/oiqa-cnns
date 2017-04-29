
'''
Extract images affected with JPEG compression artefacts.
~ Christopher Pramerdorfer
'''

import cv2

import os

src = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images'
src_imext = '.png'

dst_root = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-jpeg'
qualities = (20, 35, 50, 65, 80, 95)

# -------

assert(os.path.isdir(src))
assert(not os.path.exists(dst_root))

# -------

os.mkdir(dst_root)

for q in qualities:
    os.mkdir(os.path.join(dst_root, 'q{}'.format(q)))

# -------

ims = sorted([os.path.join(src, f) for f in os.listdir(src) if f.endswith(src_imext)])
print('{} images found'.format(len(ims)))

for imp in ims:
    print(os.path.basename(imp))

    im = cv2.imread(imp, cv2.IMREAD_UNCHANGED)
    imf = os.path.splitext(os.path.basename(imp))[0]

    for q in qualities:
        cv2.imwrite(os.path.join(dst_root, 'q{}'.format(q), '{}.jpg'.format(imf)), im, [int(cv2.IMWRITE_JPEG_QUALITY), q])

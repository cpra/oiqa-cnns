
'''
Extract images affected with Gaussian blur.
~ Christopher Pramerdorfer
'''

import cv2

import os

src = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images'
src_imext = '.png'

dst_root = '/mnt/storage/datasets/image-manipulation/RAISE/RAISE20-1300-1337/images-gblur'
sigma = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# -------

assert(os.path.isdir(src))
assert(not os.path.exists(dst_root))

# -------

os.mkdir(dst_root)

for s in sigma:
    os.mkdir(os.path.join(dst_root, 's{:.2f}'.format(s)))

# -------

ims = sorted([os.path.join(src, f) for f in os.listdir(src) if f.endswith(src_imext)])
print('{} images found'.format(len(ims)))

for imp in ims:
    print(os.path.basename(imp))

    im = cv2.imread(imp, cv2.IMREAD_UNCHANGED)
    imf = os.path.splitext(os.path.basename(imp))[0]

    for s in sigma:
        imb = cv2.GaussianBlur(im, (0, 0), s)
        cv2.imwrite(os.path.join(dst_root, 's{:.2f}'.format(s), '{}.png'.format(imf)), imb)

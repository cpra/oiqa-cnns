
This folder contains Python code for creating the RAISE dataset used in the paper. The [original RAISE dataset](http://mmlab.science.unitn.it/RAISE/) is required. The following steps are required to recreate the exact same dataset:

1. In `resize_raise.py` change the `source` and `dest` variables to point to the the original raise dataset (a folder with 8256 TIF images) and destination directory, respectively. Then execute the script using `python3 ./resize_raise.py`. It will select a random subset of 1300 images (the same subset used in the paper), scale these images to 20% and save them.
2. In `split_train_val_test.py` again change the `source` and `dest` variables. The former should point to the `dest` directory selected in step 1, the latter can be any filename. Then run the script; it will create a single JSON file that indicates to which subset (train, validation, test) each of the 1300 images belongs.
3. In `sample_patch_locations.py` change `source` to point to `dest` of step 1, `tvt` to `dest` of step 2, and `dest` to any folder that does not yet exist. Then run the script; it will randomly extract 200 patch locations for every image and save these locations to corresponding files in `dest/`. Training patches will have size 40x40, all others 32x32.
4. Create images affected with JPEG compression, JP2K compression, and blurring using the corresponding `gendata_*.py` scripts. In each one change `src` to `dest` of step 1 and `dst_root` to some not existing folder and then run it.
5. In `extract_patches.py` change `patch_locations` to `dest` from step 3, `split_file` to `dest` from step 2, and `dest` to any not existing folder. Also change the paths in the `tasks` dictionary to match up with those used in step 4. The final dataset in HDF5 format will be saved to `dest`.

The scripts have the following dependencies:

* [NumPy](http://www.numpy.org/)
* [OpenCV](http://opencv.org/) with Python wrappers
* [h5py](http://www.h5py.org/)

In order for `gendata_jp2_compression.py` to work, [ImageMagick](https://www.imagemagick.org/script/index.php) and [OpenJPEG](http://www.openjpeg.org/) must be installed. These come with `convert` and `image_to_j2k` applications, respectively.

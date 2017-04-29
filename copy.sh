
ROOT_PATH="/home/chris/projects/caip-paper"

mkdir -p gen_raise

cp "${ROOT_PATH}/scripts/RAISE/resize_raise.py" gen_raise/
cp "${ROOT_PATH}/scripts/split_train_val_test.py" gen_raise/
cp "${ROOT_PATH}/scripts/sample_patch_locations.py" gen_raise/

cp "${ROOT_PATH}/scripts/RAISE/gendata_jpeg_compression.py" gen_raise/
cp "${ROOT_PATH}/scripts/RAISE/gendata_jp2_compression.py" gen_raise/
cp "${ROOT_PATH}/scripts/RAISE/gendata_gblur.py" gen_raise/

cp "${ROOT_PATH}/scripts/RAISE-multi-clf/extract_patches.py" gen_raise/

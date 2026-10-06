# Data locations used by the run scripts. Edit these, or export them before running.
#
# H5_BASE:  image caches built by tools/build_h5_cache.py, with coco/ and tdiuc/ subfolders.
# VQA_BASE: question and answer files, with vqa/ and tdiuc/ subfolders.
# Relative paths are resolved from the repository root.

H5_BASE="${H5_BASE:-./h5_dataset}"
VQA_BASE="${VQA_BASE:-./datasets}"

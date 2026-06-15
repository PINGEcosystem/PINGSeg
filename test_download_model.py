import os
from pingseg.rf_utils import get_model

utils_dir = '/mnt/c/Users/cbodine/.test_rf_seg_download'
if not os.path.exists(utils_dir):
    os.makedirs(utils_dir)

get_model(utils_dir=utils_dir)
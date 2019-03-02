# Date: 2019.1.13
# Author: Kingdrone

from skimage.io import imread, imsave
import os
import glob
import numpy as np

def merge_image_rows_cols(img_dir, row_num, col_num, clip_size=512, img_format='png'):
    """

    Args:
        img_dir:
        img_format:

    Returns:

    """
    res_arr = np.zeros((clip_size*row_num, clip_size*col_num, 3)).astype(np.uint8)
    for i in range(row_num):
        for j in range(col_num):
            img_path = os.path.join(img_dir, "%d_%d.%s" % (i+1, j+1, img_format))
            img_arr = imread(img_path)
            res_arr[i * clip_size:(i+1)*clip_size, j * clip_size:(j+1)*clip_size, :] = img_arr
            print("[%d/%d]"%(i*col_num+j, row_num*col_num))
    imsave('merge.png', res_arr)

if __name__ == '__main__':
    img_dir = r'D:\Work_Space\shaobing\wh_predict'
    merge_image_rows_cols(img_dir, 5, 6)
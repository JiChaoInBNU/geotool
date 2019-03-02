from skimage.io import imread, imsave
import cv2
import os
import glob
import random
import numpy as np
import shutil


class SegData(object):
    def __init__(self,
                 image_dir,
                 mask_dir=None,
                 img_format='jpg',
                 mask_format='png',
                 id_txt=None,
                 im_mask_id_list=None
                 ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.im_mask_id_list = im_mask_id_list
        self.id_list = [i.strip() for i in open(id_txt)] if id_txt is not None else []
        self.im_format = img_format
        self.mask_format = mask_format
        self.img_to_mask_index = self._build_index()

    def _build_index(self):
        img_to_mask_index = []
        img_path_list = glob.glob(os.path.join(self.image_dir, '*.{}'.format(self.im_format)))

        if self.id_list is not None:
            for id in self.id_list:
                img_path = os.path.join(self.image_dir, "%s.%s" % (id, self.im_format))
                mask_path = os.path.join(self.mask_dir, "%s.%s" % (id, self.mask_format))
                # get img_to_mask_index
                img_to_mask_index.append((img_path, mask_path))

        if self.im_mask_id_list is not None:
            mask_path_list = [os.path.join(self.mask_dir, self.im_mask_id_list(os.path.split(im_path)[-1])) for
                              im_path in
                              img_path_list]
        else:

            """
            img_path_list = []
            mask_path_list = []
            for img_path in os.listdir(self.mask_dir):
              if 'label' not in img_path:
                img_path_list.append(os.path.join(self.image_dir, img_path))
              else:
                mask_path_list.append(os.path.join(self.mask_dir, img_path))

            """
            img_path_list = glob.glob(os.path.join(self.image_dir, '*.{}'.format(self.im_format)))
            mask_path_list = glob.glob(os.path.join(self.mask_dir, '*.{}'.format(self.mask_format)))

            img_path_list.sort()
            mask_path_list.sort()
        img_to_mask_index = [(img_path_list[i], mask_path_list[i]) for i in range(len(img_path_list))]
        return img_to_mask_index

    def __len__(self):
        return len(self.img_to_mask_index)

    def __getitem__(self, index):
        return self.img_to_mask_index[index]


    def resize_image_and_mask(self, save_dir, size=(512, 512), scale_factor=None):
        for idx, (img_path, mask_path) in enumerate(self.img_to_mask_index):
            im = imread(img_path)
            mask = imread(mask_path)
            if im.shape != mask.shape:
                raise AssertionError('image shape %s is not consistent with mask %s\n %s'
                                     % (im.shape, mask.shape, img_path))
            if scale_factor is not None:
                im = cv2.resize(im, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            else:
                im = cv2.resize(im, dsize=size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_NEAREST)
            img_save_path = os.path.join(save_dir, "%d.jpg" % (idx))
            mask_save_path = os.path.join(save_dir, "%d.png" % (idx))
            imsave(img_save_path, im)
            imsave(mask_save_path, mask)
            print("[%d]/[%d]%s[resize]" % (idx + 1, len(self.img_to_mask_index), img_path))

    def clip_image_and_mask(self, save_dir, size=512, start_id=0):
        os.makedirs(save_dir, exist_ok=True)
        for idx, (img_path, mask_path) in enumerate(self.img_to_mask_index):
            im = imread(img_path)
            mask = imread(mask_path)
            if im.shape != mask.shape:
                raise AssertionError('image shape %s is not consistent with mask %s\n %s'
                                     % (im.shape, mask.shape, img_path))
            h, w, d = im.shape
            row_num = int(h / size)
            col_num = int(w / size)
            for i in range(row_num):
                for j in range(col_num):
                    clipped_im = im[i * size: (i + 1) * size, j * size: (j + 1) * size]
                    clipped_mask = mask[i * size: (i + 1) * size, j * size: (j + 1) * size]
                    n = i * row_num + j
                    img_save_path = os.path.join(save_dir, "%d_%d.jpg" % (idx + start_id, n))
                    mask_save_path = os.path.join(save_dir, "%d_%d.png" % (idx + start_id, n))
                    imsave(img_save_path, clipped_im)
                    imsave(mask_save_path, clipped_mask)

            print("[%d]/[%d]%s[clip]" % (idx + 1, len(self.img_to_mask_index), img_path))

    def filter_image_and_mask(self, preserve_label_list):

        for idx, (img_path, mask_path) in enumerate(self.img_to_mask_index):
            mask = imread(mask_path)
            w, h, d = mask.shape
            mask = mask.reshape([-1, d]).tolist()
            f = False
            for l in preserve_label_list:
                if l in mask:
                    f = True
                    break
            if f is False:
                os.remove(img_path)
                os.remove(mask_path)
                print("[%d]/[%d]%s[filter]" % (idx + 1, len(self.img_to_mask_index), img_path))


    def statical_ratio(self, class_num=6):
        img_to_mask_index = self.img_to_mask_index.copy()
        ratio = np.zeros((class_num, 1))
        for idx, (_, mask_path) in enumerate(img_to_mask_index):
            mask = imread(mask_path)
            mask = process_mask_with_background(mask).reshape([-1])
            pixel_num = len(mask)
            # index = np.unique(mask)
            each_percentage = np.bincount(mask)/pixel_num
            print(mask_path)
            for i in range(len(each_percentage)):
                ratio[i] += each_percentage[i]
        print(ratio/len(img_to_mask_index))

    def convert_mask_to_gray(self, map_dict, save_dir, default_value=0):
        os.makedirs(save_dir, exist_ok=True)
        for idx, (img_path, mask_path) in enumerate(self.img_to_mask_index):
            mask = imread(mask_path)
            h, w, d = mask.shape
            gray_mask = np.zeros((h, w), np.long)
            for i in range(h):
                for j in range(w):
                    v = str(mask[i, j].tolist())
                    label = map_dict.get(v, default_value)
                    gray_mask[i, j] = label
            save_gray_mask_path = os.path.join(save_dir, os.path.basename(mask_path))
            imsave(save_gray_mask_path, gray_mask)
            print("[%d]/[%d]%s[gray_mask]" % (idx + 1, len(self.img_to_mask_index), img_path))

    def split_image_and_mask(self, split_num, val_dir):
        img_to_mask_index = self.img_to_mask_index.copy()
        random.shuffle(img_to_mask_index)
        val_num = int(len(img_to_mask_index) / split_num)
        os.makedirs(val_dir, exist_ok=True)
        for i in range(val_num):
            img_path = img_to_mask_index[i][0]
            mask_path = img_to_mask_index[i][1]
            shutil.move(img_path, os.path.join(val_dir, os.path.basename(img_path)))
            shutil.move(mask_path, os.path.join(val_dir, os.path.basename(mask_path)))
            print("[%d]/[%d]%s[move]" % (i + 1, val_num, img_path))


import threading

exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, thread_id, func, args):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.func = func
        self.args = args

    def run(self):
        print("Starting " + self.thread_id)
        self.func(*self.args)
        print("Exiting " + self.thread_id)


def process_mask_with_background(mask):
    convert_mask = np.zeros(mask.shape[:2]).astype(np.long)
    # label
    convert_mask[(mask[:, :, 0] < 128) * (128 < mask[:, :, 1]) * (mask[:, :, 2] > 128)] = 0  # Urban Land
    convert_mask[(mask[:, :, 0] > 128) * (mask[:, :, 1] > 128) * (128 > mask[:, :, 2])] = 1  # Agriculture land
    convert_mask[(mask[:, :, 0] > 128) * (mask[:, :, 1] < 128) * (128 < mask[:, :, 2])] = 2  # Range Land
    convert_mask[(mask[:, :, 0] < 128) * (128 < mask[:, :, 1]) * (mask[:, :, 2] < 128)] = 3  # Forest land
    convert_mask[(mask[:, :, 0] < 128) * (mask[:, :, 1] < 128) * (128 < mask[:, :, 2])] = 4  # Water
    convert_mask[(mask[:, :, 0] > 128) * (mask[:, :, 1] > 128) * (mask[:, :, 2] > 128)] = 5  # Barren land

    return convert_mask

def compute_mean_std(img_dir, format='jpg'):
    img_list = glob.glob(os.path.join(img_dir, '*.{}'.format(format)))
    sum_list = [0, 0, 0]
    total_pixel_num = 0
    for idx, img_path in enumerate(img_list):
        im = imread(img_path)
        d = im.shape[-1]
        im = im.reshape([-1, d])
        total_pixel_num += im.shape[0]
        s = np.sum(im, axis=0)
        sum_list = [sum_list[i]+s[i] for i in range(3)]
        print("[%d/%d]%s" % (idx, len(img_list),img_path))
        print(sum_list)
    mean = [(i / total_pixel_num) for i in sum_list]
    print('mean: %s' % (mean))

    sum_var_list = [0, 0, 0]
    # mean = np.array([104.09483293, 96.66852301, 71.80584479])
    for idx, img_path in enumerate(img_list):
        im = imread(img_path)
        im = im.reshape([-1, 3])
        total_pixel_num += im.shape[0]
        var = np.sum((im - mean)**2, axis=0)
        sum_var_list = [sum_var_list[i] + var[i] for i in range(3)]
        print("[%d/%d]%s" % (idx, len(img_list), img_path))
        print(sum_var_list)
    var = [(i / total_pixel_num)**0.5 for i in sum_var_list]
    print('mean:%s' % mean)
    print('var: %s' % var)
    return mean , var

if __name__ == '__main__':
    img_dir = r'D:\Work_Space\PyProject\land-train\land-train'
    # sd = SegData(img_dir, img_dir)
    compute_mean_std(img_dir)


    # img_dir = r'C:\Users\Kingdrone\Desktop\data\train'
    # seg_data = SegData(img_dir, img_dir)
    # val_dir = r'C:\Users\Kingdrone\Desktop\data\1'
    # seg_data.split_image_and_mask(2, val_dir=r'D:\Work_Space\PyProject\land-train\4')
    """
    threads = []
    for i in range(1, 11):
        start_id = (i - 1) * 15 + 805
        img_dir = r'/mnt/disk2/wjjdata/GID/' + str(i)
        seg_data = SegData(img_dir, img_dir, 'tif', 'tif')
        args = (r'/mnt/disk2/wjjdata/GID/clipped_' + str(i), 512, start_id)
        thread = myThread(str(i), seg_data.clip_image_and_mask, args)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
    print("Exiting Main Thread")
    """

    """
    threads = []
    for i in range(1, 11):
      img_dir = '/mnt/disk2/wjjdata/GID/clipped_' + str(i)
      seg_data = SegData(img_dir, img_dir, 'jpg', 'png')
      args = ([[0, 0, 255]], )
      thread = myThread(str(i), seg_data.filter_image_and_mask, args)
      thread.start()

    for t in threads:
        t.join()
    print("Exiting Main Thread")
    """
    """
    threads = []
    for i in range(1, 11):
        img_dir = '/mnt/disk2/wjjdata/GID/clipped_' + str(i)
        save_dir = '/mnt/disk2/wjjdata/GID/gray_mask' + str(i)
        seg_data = SegData(img_dir, img_dir, 'jpg', 'png')
        map_dict = {'[0, 0, 255]': 2}
        args = (map_dict, save_dir)
        thread = myThread(str(i), seg_data.convert_mask_to_gray, args)
        thread.start()

    for t in threads:
        t.join()
    print("Exiting Main Thread")
    """






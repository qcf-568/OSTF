import pickle
import json
import cv2
import os
import random
import numpy as np

# def check_box(box):
#     if box:
#         for b in box:
#             if b[0]==b[2] or b[1]==b[3]:
#                 return False
#         return True
#     else:
#         return True

# def transfer(train_imgs_dir,train_list):
#     train = {}
#     for img in train_list:
#         box = img['region']
#         check = check_box(box)
#         if check:
#             img_path = os.path.join(train_imgs_dir,img['id'])
#             v = {}
#             image = cv2.imread(img_path)
#             v['h'] = image.shape[0]
#             v['w'] = image.shape[1]
#             v['b'] = np.array(box,dtype=int)
#             train[img_path] = v
#     return train


# train_imgs_dir = 'datas/afac2023/train/train_images'

# with open('datas/afac2023/train/label_train.json') as f:
#     data = json.load(f)

# random.shuffle(data)
# split_index = int(0.9 * len(data))
# train_list = data[:split_index]
# val_list = data[split_index:]

# train = transfer(train_imgs_dir,train_list)
# val = transfer(train_imgs_dir,val_list)
# with open('datas/afac2023/train/train.pk', 'wb') as f:
#     pickle.dump(train, f)

# with open('datas/afac2023/train/val.pk', 'wb') as f:
#     pickle.dump(val, f)



# 从 PK 文件中加载对象
with open('datas/afac2023/train/train.pk', 'rb') as f:
    train = pickle.load(f)
with open('datas/afac2023/train/val.pk', 'rb') as f:
    val = pickle.load(f)


def pk_transfer(loaded_data,new):
    train = {}
    for path in loaded_data:
        new_path = path.replace('train_images',new)
        train[new_path] = loaded_data[path]
    return train

save = 'train_masks'
train_mask = pk_transfer(train, save)
with open('datas/afac2023/train/train_masks.pk', 'wb') as f:
    pickle.dump(train_mask, f)

val_mask = pk_transfer(val, save)
with open('datas/afac2023/train/val_masks.pk', 'wb') as f:
    pickle.dump(val_mask, f)


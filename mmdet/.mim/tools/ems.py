import os
import json
import cv2
from tqdm import tqdm

json_name = 'results/swin/exp11_ep97_0.5.json'
imgs_dir = 'datas/afac2023/val_images'
visual_dir = 'results/swin/exp11_ep97_0.5'

if not os.path.exists(visual_dir):
    os.makedirs(visual_dir)

with open(json_name,'r') as file:
    data = json.load(file)
for info in tqdm(data):
    img_name = info['id']
    boxes = info['region']
    if boxes:
        box = boxes[0]
        imgs = cv2.imread(os.path.join(imgs_dir,img_name))
        cv2.rectangle(imgs,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
        cv2.imwrite(os.path.join(visual_dir,img_name),imgs)
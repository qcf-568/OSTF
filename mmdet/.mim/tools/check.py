import pickle
import json
import cv2
import os
import random
import numpy as np

with open('datas/afac2023/train/train.json') as f:
    data = json.load(f)

for inf in data:
    box = inf['region']
    if box:
        for b in box:
            if b[0]==b[2] or b[1]==b[3]:
                print(inf)
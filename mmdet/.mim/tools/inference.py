from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
import torch
import os
import json
from tqdm import tqdm
import cv2
from sklearn.decomposition import PCA
from mmengine import ConfigDict
from mmengine.config import Config



def pca(img):
    n=1
    pca = PCA(n_components=n)
    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)

    h,w,c = img.shape
    img_reshape = img.reshape(h*w,c)

    img_pca = pca.fit_transform(img_reshape)
    result = 255-img_pca.reshape(h,w,n)

    result = 0.5 * result + 0.5 * img
    return img

def mask2pca(i,mask):
    img = cv2.imread(os.path.join('datas/afac2023/val_images',i))
    n=1
    pca = PCA(n_components=n)
    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)

    h,w,c = img.shape
    img_reshape = img.reshape(h*w,c)

    img_pca = pca.fit_transform(img_reshape)
    result = 255-img_pca.reshape(h,w,n)

    result = 0.5 * result + 0.5 * mask
    result = cv2.convertScaleAbs(result)

    result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    return img  

def mask2img(i,mask):
    img = cv2.imread(os.path.join('datas/afac2023/val_images',i))
    img = 0.5*img + 0.5*mask
    result = cv2.convertScaleAbs(result)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

imgs_dir = 'datas/afac2023/val_images'
# imgs_dir = 'datas/afac2023/val_convnext_predict_mask_exp4_ep124'
# imgs_dir = 'datas/afac2023/val_convnext_predict_mask_exp5_ep50'

config_file = './work_configs/cascade_rcnn_swin_fpn_2fc.py'
# config_file = 'work_configs/rtm.py'
# checkpoint_file = 'work_dirs/rtm_mask/exp1/best_ACC_hmean_epoch_50.pth'
# config_file = './work_configs/cascade_rcnn_convnext_fpn_2fc.py'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp3/best_ACC_hmean_epoch_98.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp7_masks/best_ACC_hmean_epoch_57.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp8_mask2img/best_ACC_hmean_epoch_52.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp7_masks/epoch_100.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp8_mask2img/best_ACC_hmean_epoch_82.pth'
checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp11/best_ACC_hmean_epoch_118.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp6/best_ACC_hmean_epoch_35.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_convnext_fpn_2fc/exp1/best_ACC_hmean_epoch_116.pth'
# checkpoint_file = 'work_dirs/cascade_rcnn_swin_fpn_2fc/exp4/best_ACC_hmean_epoch_107.pth'


register_all_modules()
# build the model from a config file and a checkpoint file
config_file = Config.fromfile(config_file)
config_file.model = ConfigDict(**config_file.tta_model, module=config_file.model)
test_data_cfg = config_file.test_dataloader.dataset
while 'dataset' in test_data_cfg:
    test_data_cfg = test_data_cfg['dataset']
# batch_shapes_cfg will force control the size of the output image,
# it is not compatible with tta.
if 'batch_shapes_cfg' in test_data_cfg:
    test_data_cfg.batch_shapes_cfg = None
test_data_cfg.pipeline = config_file.tta_pipeline
model = init_detector(config_file, checkpoint_file, device='cuda:0',cfg_options={})

imgs = os.listdir(imgs_dir)
imgs = sorted(imgs)
l = len(imgs)-1
num = 0
with open('results/swin/exp11_ep97_0.5.json', "w", encoding="utf-8") as json_file:
    json_file.write("[\n")
    for iter,i in enumerate(tqdm(imgs)):
        path = os.path.join(imgs_dir,i)
        # print(i)
        # img = mmcv.imread( path, channel_order='rgb')
        img = cv2.imread(path)
        # img = mask2pca(i,img)  #mask2pca
        # img = mask2img(i,img) #mask2img

        result = inference_detector(model, img)

        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes
        scores = pred_instances.scores
        bboxes = bboxes.tolist()
        scores = scores.tolist()
        if bboxes:
            if max(scores)>=0.5:
                num += 1
                bboxes = [bboxes[scores.index(max(scores))]]
                bboxes = [[round(coord, 1) for coord in bbox] for bbox in bboxes]
            else:
                bboxes = []
        d = {}
        d['id'] = i
        d['region'] = bboxes
        # d['score'] = scores
        json_file.write("\t")
        json.dump(d, json_file)
        if iter == l:
            json_file.write("\n]")
        else:
            json_file.write(",\n")
print('检测出篡改的图片数量：',num)
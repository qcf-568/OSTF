# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
from typing import List, Optional, Sequence, Tuple, Union
import albumentations
import os
import cv2
import mmcv
import numpy as np
import albumentations.augmentations.functional as F
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale

from PIL import Image
from numba import njit, prange
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]

@TRANSFORMS.register_module()
class Texture2(BaseTransform):
    def __init__(self, prob=1.0, bathres=2048, revjpegpath='', diffbirpath='', diffstepath='', q1=75, q2=100, rgbshift=20) -> None:
        assert 0 <= prob <= 1
        self.q1 = q1
        self.q2 = q2
        self.p = prob
        self.bathres = bathres
        self.rgbshift = rgbshift
        self.diffstepath = diffstepath
        self.diffbirpath = diffbirpath
        self.revjpegpath = revjpegpath
        self.kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.int32)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        h,w = results['img_shape']
        whs = results['gt_bboxes_whs']
        # print(results['gt_bboxes_whs'], h, w)
        # exit(0)
        img1 = results['img']
        h,w = img1.shape[:2]
        img2 = img1.copy()
        # print(results.keys(),results['gt_bboxes'].shape, results['gt_bboxes_labels'].shape,results['img'].shape)
        results['gt_bboxes_tamper_type0'] = np.full((len(results['gt_bboxes_labels']),),-1,dtype=np.int32)
        results['gt_bboxes_tamper_type1'] = np.full((len(results['gt_bboxes_labels']),),-1,dtype=np.int32)
        results['gt_bboxes_tamper_same'] = np.full((len(results['gt_bboxes_labels']),),-1,dtype=np.int32)
        tamps=0
        lens = len(results['gt_bboxes'])
        nospe = (not os.path.exists(os.path.join(self.revjpegpath, results['img_path'])))
        this_canvas1 = np.zeros((h,w), dtype=np.float32)
        this_canvas2 = this_canvas1.copy()
        shuffle_inds = {k:v for k,v in enumerate(np.random.choice(lens, lens, replace=False))}
        for bnum in range(lens):#enumerate(results['gt_bboxes']):
            ### too small, type -1 to be ignored
            bi = shuffle_inds[bnum]
            box = results['gt_bboxes'][bi]
            bh, bw, ba = results['gt_bboxes_whs'][bi].astype(np.int32)
            if ((bh<32) or (bw<32) or (whs[bi][2]<=self.bathres) or nospe):
                continue
            box = box.numpy().squeeze().astype(np.int32)
            y1,x1,y2,x2 = box
            if ((y2>w) or (x2>h) or (y1>=y2) or (x1>=x2)):
                continue
            ### authentic, type0
            if tamps>64:
                results['gt_bboxes_tamper_type0'][bi] = 0
                results['gt_bboxes_tamper_type1'][bi] = 0
                results['gt_bboxes_tamper_same'][bi] = 0
                continue
            bws = bw//2
            bhs = bh//2
            xmin = int(max(0, x1-bhs))
            xmax = int(min(h, x2+bhs))
            ymin = int(max(0, y1-bws))
            ymax = int(min(w, y2+bws))
            newbw = (ymax-ymin)
            newbh = (xmax-xmin)
            flag = 0
            if (this_canvas1[x1:x2,y1:y2].sum()==0):
                if (random.uniform(0,1)>0.5):
                    tamptype = self.tamper(img1, y1, x1, y2, x2, bh, bw, bhs, bws, xmin, xmax, ymin, ymax, newbw, newbh, h, w, results['gt_bboxes_pts'][bi], results, results['gt_bboxes_whs'][bi])
                    results['gt_bboxes_tamper_type0'][bi] = tamptype
                    this_canvas1[x1:x2,y1:y2] = 1
                    tamps = (tamps+1)
                    flag = 2
                else:
                    results['gt_bboxes_tamper_type0'][bi] = 0
                    this_canvas1[x1:x2,y1:y2] = 1
                    flag = 1
            if (this_canvas2[x1:x2,y1:y2].sum()==0):
                if (random.uniform(0,1)>0.5):
                    tamptype = self.tamper(img2, y1, x1, y2, x2, bh, bw, bhs, bws, xmin, xmax, ymin, ymax, newbw, newbh, h, w, results['gt_bboxes_pts'][bi], results, results['gt_bboxes_whs'][bi])
                    results['gt_bboxes_tamper_type1'][bi] = tamptype
                    # results['gt_bboxes_labels1'][bi] = 1
                    this_canvas2[x1:x2,y1:y2] = 1
                    tamps = (tamps+1)
                    if (flag==2):
                        results['gt_bboxes_tamper_same'][bi] = 2 # 2 all_tamper
                    elif (flag==1):
                        results['gt_bboxes_tamper_same'][bi] = 1 # 1 one_tamper
                else:
                    results['gt_bboxes_tamper_type1'][bi] = 0
                    this_canvas2[x1:x2,y1:y2] = 1
                    if (flag==2):
                        results['gt_bboxes_tamper_same'][bi] = 1 # 1 one_tamper
                    elif (flag==1):
                        results['gt_bboxes_tamper_same'][bi] = 0 # 0 all_authentic
        rs = random.randint(-self.rgbshift, self.rgbshift)
        gs = random.randint(-self.rgbshift, self.rgbshift)
        bs = random.randint(-self.rgbshift, self.rgbshift)
        img1 = F.shift_rgb(img1, rs, gs, bs)
        img2 = F.shift_rgb(img2, rs, gs, bs)
        img1 = cv2.imdecode(cv2.imencode('.jpg',img1,[1,random.randint(self.q1,self.q2)])[1],1)
        img2 = cv2.imdecode(cv2.imencode('.jpg',img2,[1,random.randint(self.q1,self.q2)])[1],1)
        results['img'] = np.concatenate((img1, img2), 2)
        assert results['img'].shape[2]==6
        return results

    def tamper(self, img, y1, x1, y2, x2, bh, bw, bhs, bws, xmin, xmax, ymin, ymax, newbw, newbh, h, w, pts, results, bhw):
        # box = box.numpy().squeeze().astype(np.int32)
        canvas = np.zeros((newbh, newbw), dtype=np.float32)
        if (len(pts)>=4):
            newpts = np.stack((pts[:,0]-ymin, pts[:,1]-xmin), 1).astype(np.int32)
            cv2.fillPoly(canvas, [newpts], 1)
        else:
            newx1 = (x1-xmin)
            newy1 = (y1-ymin)
            newx2 = (x2-xmin)
            newy2 = (y2-ymin)
            canvas[newx1:newx2, newy1:newy2] = 1
        dilateh = bh//2
        dilatew = bw//4
        if (dilateh%2==0):
            dilateh = (dilateh+1)
        if (dilatew%2==0):
            dilatew = (dilatew+1)
        # print(results['img_path'], dilateh, dilatew)
        canvas = cv2.GaussianBlur(canvas, (dilateh, dilatew), dilateh/8.0, dilatew/8.0)
        canvas = canvas[...,None]
        # print('imgshape', img.shape, xmin, xmax, ymin, ymax, x1, x2, y1, y2, results['img_path'], img[xmin:xmax, ymin:ymax].shape)
        img_tamp, tamper_type = self.img_tamper(img[xmin:xmax, ymin:ymax].copy(), bh, xmin, ymin, xmax, ymax, results)
        # print('shapes',img_tamp.shape, canvas.shape, img[xmin:xmax, ymin:ymax].shape)
        img[xmin:xmax, ymin:ymax] = ((img_tamp.astype(np.float32))*canvas+(img[xmin:xmax, ymin:ymax].astype(np.float32))*(1-canvas)).astype(np.uint8)
        return tamper_type

    def img_tamper(self, img, bh, xmin, ymin, xmax, ymax, results):
       rand = random.uniform(0,1)
       exists_revjpeg = os.path.exists(os.path.join(self.revjpegpath, results['img_path']))
       exists_diffste = os.path.exists(os.path.join(self.diffstepath, results['img_path']))
       exists_diffbir = os.path.exists(os.path.join(self.diffbirpath, results['img_path']))
       if (rand<0.35):
           if exists_revjpeg:
               return (self.revjpeg(xmin, ymin, xmax, ymax, results), 2)
           else:
               return (self.blur(img, bh), 1)
       elif (rand<0.7):
           if exists_diffste:
               return (self.diffste(xmin, ymin, xmax, ymax, results), 3)
           elif exists_revjpeg:
               return (self.revjpeg(xmin, ymin, xmax, ymax, results), 2)
           else:
               return (self.blur(img, bh), 1)
       else:
           if exists_diffbir:
               return (self.diffbir(xmin, ymin, xmax, ymax, results), 4)
           elif exists_revjpeg:
               return (self.revjpeg(xmin, ymin, xmax, ymax, results), 2)
           else:
               return (self.blur(img, bh), 1)

    def jpeg(self, img, bh):
        if bh<=16:
            return cv2.imdecode(cv2.imencode('.jpg',img,[1,random.randint(40,70)])[1],1)
        elif bh<=64:
            return cv2.imdecode(cv2.imencode('.jpg',img,[1,random.randint(30,65)])[1],1)
        elif bh<=256:
            return cv2.imdecode(cv2.imencode('.jpg',img,[1,random.randint(20,60)])[1],1)
        else:
            return cv2.imdecode(cv2.imencode('.jpg',img,[1,random.randint(10,50)])[1],1)

    def revjpeg(self, xmin, ymin, xmax, ymax, results):
        return cv2.imread(os.path.join(self.revjpegpath, results['img_path']))[xmin: xmax, ymin: ymax]

    def diffbir(self, xmin, ymin, xmax, ymax, results):
        return cv2.imread(os.path.join(self.diffbirpath, results['img_path']))[xmin: xmax, ymin: ymax]

    def diffste(self, xmin, ymin, xmax, ymax, results):
        return cv2.imread(os.path.join(self.diffstepath, results['img_path']))[xmin: xmax, ymin: ymax]

    def blur(self, img, bh):
        rand = random.uniform(0,1)
        if rand<0.3:
            return self.gauss_blur(img, bh)
        elif rand<0.6:
            return self.pixelate(img, bh)
        elif rand<0.9:
            return self.defocus_blur(img, bh)
        else:
            return self.motion_blur(img, bh)

    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)
        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def getOptimalKernelWidth1D(self, radius, sigma):
        return radius * 2 + 1

    def gauss_function(self, x, mean, sigma):
        return (np.exp(- (x - mean)**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)

    def getMotionBlurKernel(self, width, sigma):
        k = self.gauss_function(np.arange(width), 0, sigma)
        Z = np.sum(k)
        return k/Z

    def shift(self, image, dx, dy):
        if(dx < 0):
            shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
            shifted[:,dx:] = shifted[:,dx-1:dx]
        elif(dx > 0):
            shifted = np.roll(image, shift=dx, axis=1)
            shifted[:,:dx] = shifted[:,dx:dx+1]
        else:
            shifted = image
        if(dy < 0):
            shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
            shifted[dy:,:] = shifted[dy-1:dy,:]
        elif(dy > 0):
            shifted = np.roll(shifted, shift=dy, axis=0)
            shifted[:dy,:] = shifted[dy:dy+1,:]
        return shifted

    def _motion_blur(self, x, radius, sigma, angle):
        width = self.getOptimalKernelWidth1D(radius, sigma)
        kernel = self.getMotionBlurKernel(width, sigma)
        point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
        hypot = math.hypot(point[0], point[1])
        blurred = np.zeros_like(x, dtype=np.float32)
        for i in range(width):
            dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
            dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
            if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
                # simulated motion exceeded image borders
                break
            shifted = self.shift(x, dx, dy)
            blurred = blurred + kernel[i] * shifted
        return blurred

    def gauss_blur(self, x, h):
        if h<=100:
            bh = int(h**(1/2))
        else:
            bh = random.randint(6,12)
        if (bh%2==0):
            bh = (bh+1)
        return cv2.GaussianBlur(x, (bh, bh), 0)

    def motion_blur(self, x, h):
        if h<=16:
            c = (random.randint(8,10), random.uniform(2,3))
        elif h<=64:
            c = (random.randint(9,11), random.uniform(2.5,3.5))
        elif h<=256:
            c = (random.randint(10,12), random.uniform(3,4))
        else:
            c = (random.randint(11,14), random.uniform(4,5))
        shape = x.shape
        angle = np.random.uniform(-45, 45)
        x = self._motion_blur(x, radius=c[0], sigma=c[1], angle=angle)
        if len(x.shape) < 3 or x.shape[2] < 3:
            gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
            if len(shape) >= 3 or shape[2] >= 3:
                return np.stack([gray, gray, gray], axis=2)
            else:
                return gray
        else:
            return np.clip(x, 0, 255)

    def defocus_blur(self, x, h):
        if h<=8:
            c = (random.uniform(1.2,1.4), 0.1)
        elif h<=16:
            c = (random.uniform(1.25,1.6), 0.1)
        elif h<=32:
            c = (random.uniform(1.3,1.8), 0.1)
        elif h<=64:
            c = (random.uniform(1.4,2.0), 0.1)
        elif h<=128:
            c = (random.uniform(1.5,2.2), 0.1)
        else:
            c = (random.uniform(1.6,2.4), 0.1)
        x = (x / 255.)
        kernel = self.disk(radius=c[0], alias_blur=c[1])
        channels = []
        if len(x.shape) < 3 or x.shape[2] < 3:
            channels = np.array(cv2.filter2D(x, -1, kernel))
        else:
            for d in range(3):
                channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
            channels = np.array(channels).transpose((1, 2, 0))
        return np.clip(channels, 0, 1) * 255

    def pixelate(self, x, h):
        if h<=8:
            c = random.uniform(0.8, 0.6)
        elif h<=16:
            c = random.uniform(0.75, 0.5)
        elif h<=32:
            c = random.uniform(0.7, 0.4)
        elif h<=64:
            c = random.uniform(0.65, 0.3)
        elif h<=128:
            c = random.uniform(0.6, 0.2)
        elif h<=256:
            c = random.uniform(0.55, 0.15)
        else:
            c = random.uniform(0.5, 0.1)
        x_shape = x.shape
        x = cv2.resize(x, (int(x_shape[1] * c), int(x_shape[0] * c)), 0)
        x = cv2.resize(x, (x_shape[1], x_shape[0]), random.randint(0,3))
        return x

    def revblur(self, img):
        return cv2.filter2D(img,-1,kernel=self.kernel)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        return repr_str

# Revisiting Tampered Scene Text Detection in the Era of Generative AI [AAAI2025]

### This is the official implementation of the paper Revisiting Tampered Scene Text Detection in the Era of Generative AI.  [paper](https://arxiv.org/pdf/2407.21422)

---

### The Open-Set Text Forensics (OSTF) dataset is now publicly available at [Google Drive](https://drive.google.com/file/d/16Pyv7nLBOsOefwzdCsa0ndXxnzknfxtw/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/10FbI3SfWWV92vqv3X-ILxQ?pwd=OSTF). 

<font size=10>Researchers are welcome 😃 to apply for this dataset by sending an email to  202221012612@mail.scut.edu.cn (with institution email address) and introducing:</font><br/>
1. Who you are and your institution.
2. Who is your supervisor/mentor.
---
### OSTF Train data preparation
1. Apply, download and unzip the OSTF dataset.
2. Move all the 18 *.pk files from the mmacc_pks dir into the mmacc dir.
3. Move the mmacc dir into this main dir.
Finally, after the above 3 steps, in this main dir, you will get such dir structre:
```
FBCNN---...
  |
configs---...
  |
mmcv_custom---...
  |
mmdet---...
  |
tools---...
  |
mmacc---srnet---...
          |
        srnet_train.pk
          |
        srnet_test.pk
          |
        anytext---...
          |
        anytext_train.pk
          |
        anytext_test.pk
          |
         ...
```
---
### Texture Jitter train data preparation
1. Download and unzip the [pretrain_pk.zip](https://drive.google.com/file/d/1xvu82bZvgq7TBXEjByFvuGi6th5ifsHY/view?usp=sharing) in this dir. After unzip, you will get a new dir named "pretrain" with 7 sub-dirs (ArT, ICDAR2013, ICDAR2015, ICDAR2017-MLT, LSVT, ReCTS, TextOCR).
2. Download and unzip the [msk.zip](https://drive.google.com/file/d/1KIcuNZ-9QRRtnPsXRkLWbaOjpyCtNHr6/view?usp=sharing) in the new "pretrain" dir. After unzip, you will get 7 new dirs all named 'msk' under the above 7 sub-dirs. 
3. Download the dataset ***training set images*** from [ArT](https://rrc.cvc.uab.es/?ch=14&com=introduction), [ICDAR2013 (Task 2.4: End to End (2015 edition))](https://rrc.cvc.uab.es/?ch=2&com=introduction), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction), [ICDAR2017-MLT](https://rrc.cvc.uab.es/?ch=8&com=introduction), [LSVT (train_full_images_0/1.tar.gz 4.1G)](https://rrc.cvc.uab.es/?ch=16&com=introduction), [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=introduction), [TextOCR](https://textvqa.org/textocr/dataset/).
4. Rename the 7 downloaded image dirs into an "img" dir under the 7 sub-dirs. For example, "mv [Your downloaded ArT train images] pretrain/ArT/img" and "mv [Your downloaded ReCTS train images] pretrain/ReCTS/img".
5. Make a new dir named "revjpegs" in this main dir, and make "pretrain" dir and sub-dirs to make sure that the dir "revjpegs" has the same sub-dir structure as the "pretrain" dir. For example, it should has the dirs "revjpegs/pretrain/ArT/img" and "revjpegs/pretrain/ReCTS/img", etc, corresponding to "pretrain/ArT/img" and "pretrain/ReCTS/img" respectively.
6. Download the fbcnn_color.pth following this [Readme.md](https://github.com/qcf-568/OSTF/tree/main/FBCNN#readme). In the [FBCNN dir](https://github.com/qcf-568/OSTF/tree/main/FBCNN), run the command to create reverse jpeg images for each of the 7 sub-dir images of the pretrain dir. For example, run "CUDA_VISIBLE_DEVICES=0 python app.py --inp pretrain/ArT/img/ --out revjpegs/ArT/img/" and "CUDA_VISIBLE_DEVICES=0 python app.py --inp pretrain/ReCTS/img/ --out revjpegs/ReCTS/img/".

Finally, after the above 5 steps, in this main dir, you will get such dir structre:
```
FBCNN---...
  |
configs---...
  |
pretrain---ArT---img---....
  |         |     |
  |         |   train.pk
  |         |
  |        ICDAR2015---img---...
  |         |           |
  |         |         train.pk
  |         |
  |        ...
  |
revjpeg---pretrain---ArT---img---....
  |                   |     |
  |                   |   train.pk
  |                   |
  |                   ICDAR2015---img---...
  |                   |           |
  |                   |         train.pk
  |                   |
  |                  ...
  |
mmcv_custom---...
  |
mmdet---...
  |
tools---...
  |
mmacc---...
```
#### The Texture Jitter method is implemented as "TextureSG" in "txt_pipeline" of the config files (e.g. [here](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py#L267)), its source code is [here](https://github.com/qcf-568/OSTF/blob/main/mmdet/datasets/transforms/transforms.py#L344). The key function for the Texture Jitter method is the function ["img_tamper" in Line450](https://github.com/qcf-568/OSTF/blob/main/mmdet/datasets/transforms/transforms.py#L450).

#### The DAF framework is implemented as [DFPNCMap3](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/dfpn_cmap3.py#L28) and [CascadeCMap3](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/cascade_cmap3.py#L46) for Faster R-CNN and Cascade R-CNN respectively.

DAF key implementation (take the [Faster R-CNN based DAF](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/dfpn_cmap3.py#L28) as an example): 
1. [Line47 Authentic Kernel implementation](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/dfpn_cmap3.py#L47). The variable "self.sgl" implements the Authentic Kernel ([the variable "self.C" in Line17](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/single_center_loss.py#L17)) and its loss function ([this forward function in Line21](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/single_center_loss.py#L21)).
2. [Line379 Authentic Kernel Modulation](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/dfpn_cmap3.py#L379) implements the modulation between te Authentic Kernel (the variable "self.sgl.C") and the global features (the variable "gloabl_feats"). In this line, the resulting variable "gloabl_feats" is the modulated authentic kernel.
3. [Line324 Training model to learn real/fake classification with feature difference](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/dfpn_cmap3.py#L324). During training, the ***feature difference*** between each RoI vector (the variable "mskf" in this line) and the modulated authentic kernel (the variable "glb_feats" in this line) is obtained by "mskf - glb_feats[gt_valid]". Then, this feature difference vector is fed into a fully-connected layer for final real/fake prediction as "self.fc(mskf - glb_feats[gt_valid])". During training, in this Line324, the loss between model prediction "self.fc(mskf - glb_feats[gt_valid])" and the ground-truth "gt_label[gt_valid].long()" is calculated to help the model learn real/fake classification.
4. [Line548 Model predicts real/fake with feature difference](https://github.com/qcf-568/OSTF/blob/main/mmdet/models/roi_heads/dfpn_cmap3.py#L548). In this line, the modulated authentic kernel is the variables "g" and "glb_feats", the input RoI feature vectors are "m" and "mask_feats". The feature difference is obtained by "(self.convert(m)-g)" and the final classification score is obtained by feeding it to the final binary classifier and softmax layer F.softmax(self.fc(self.convert(m)-g),1).
---
### Train

Enviroment based on Python 3.9.12

pip install -r requirements.txt

```
bash tools/dist_train.sh [your config_py_file] [Your gpu number, should be consistent with the dist_train.sh]
```
#### About the config files
The config files are in the [config dir](https://github.com/qcf-568/OSTF/tree/main/configs), all the config files following such name roles: ModelType_AblationType+TrainData.py. For the AblationType, o is for the original one without Texture Jitter Pre-training and x is the one with the Texture Jitter Pre-training. For example, fasterrcnn_xsrnet is the Faster R-CNN model pre-trained with Texture Jitter, and fine-tuned with the SR-Net training data and the Texture Jitter method.

#### Use the config files in the pre-training stage 
Given a config file that contains the model defination you want to pretrain (e.g. cascade R-CNN):
1. Modify the datasets line of the train dataloader, delete the fine-tune data. For example, in [cascade_xsrnet.py](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py), modify the [Line435](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py#L435) from "datasets = [ptdatas, ftdatas]," into "datasets = [ptdatas,],".
2. Modify the datasets line of the "pt_data". For example, in [cascade_xsrnet.py](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py), modify the [Line412](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py#L412) from "datasets = [ic13,ic15,ic17]," into "datasets = [ic13,ic15,ic17,art,rects,lsvt,textocrpt],".
3. Modify the pre-trained weight to initialize. We use the official COCO-pretrained backbone and detection modules (RPN, RoI Heads) for initialization in the pre-training stage. For example, in [cascade_xsrnet.py](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py), modify the [Line602](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py#L602) from 'cascade.pth' into your initial weights path (e.g. "rcnn_swin.pth"): The initial weights for Cascade R-CNN can be downloaded from [my another repo](https://github.com/qcf-568/Det_Baseline) ("rcnn_swin.pth" in the baseline zip file), the initial weights for Faster R-CNN can be downloaded from [here](https://drive.google.com/file/d/17EvozLczCyP64NxwL9Bx9kelpNZGZGPU/view?usp=sharing).

#### Use the config files in the fine-tuning stage 
Just need to modify the pre-trained weight into your pre-trained weights, the official ones are [here](https://drive.google.com/file/d/1ezb6ox-nlDk1UspCYGsSqSeSlH6DdMvC/view?usp=sharing). For example, in [cascade_xsrnet.py](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py), modify the [Line602](https://github.com/qcf-568/OSTF/blob/main/configs/cascade_xsrnet.py#L602) from 'cascade.pth' into your pre-trained weights.

---
### Tiny training implement
[Here](https://pan.baidu.com/s/12Ct3jUtoqLhVFkabST1Njg?pwd=3dvz) is a tiny implement of the training code and prepared data for fast try. The playground training and test data are all prepared. This is the Cascade R-CNN training only with the Texture Jitter ICDAR2013 and tested on Tampered IC13. To run the code only needs 3 steps:
1. Download and unzip the file. In the new dir, rename the [rcnn_swin.pth](https://github.com/qcf-568/Det_Baseline) ("rcnn_swin.pth" in the baseline zip file) to'cascade.pth'.
2. Modify the tools/dist_train.sh "CUDA_VISIBLE_DEVICES=6,7" to your own GPU ids.
3. run "bash tools/dist_train.sh cascade_debug.py 2"
---

### Pre-trained models
I have kept almost all the trained models, but the google drive space is not enough to hold all of them. So I provide the Texture Jitter pre-trained models and SR-Net fine-tuned models of Cascade R-CNN and Faster R-CNN being trained with our methods in [this file](https://drive.google.com/file/d/1ezb6ox-nlDk1UspCYGsSqSeSlH6DdMvC/view?usp=sharing). If you need more model weights, you can concat me to get them via educational email.

---
### Inference

```"mv mmdet mmdet_train; unzip mmdet_test.zip```

Then modify the config file you used in training:
1. Find the line with "max_per_img=100)))" and modify this into "max_per_img=1000)))".
2. Find the line with "FTIC15" and modify this into "FTIC15PK"
3. (Optional) Find the line "datasets = [test_srnet, test_stefann, test_mostel, test_derend, test_diffste, test_anytext, test_udifftext],", and modify it into your test dataset (may be a single one such as 'datasets = test_srnet,').

```bash tools/dist_test.sh [your config file] [model weights to evaluate] 1```

Then you will get a new .pk file in a newly created dir named "results"

#### The offcial mmdetection inference code also applies to this repo. 

### Evaluation
After inference, the model prediction is converted into .txt files, zipped and evaluated following the same [official Tampered-IC13 evaluation tools and methods](https://github.com/wangyuxin87/Tampered-IC13).

---
### Any bug or question please open an issue or concat me with email.

---
### License

The project is under [CC-BY-NC-4.0](https://www.creativecommons.org/licenses/by-nc/4.0/deed.en) license.

---
### Ackownledgement

The **SRNet** part of the OSTF dataset is directly borrowed from the [Tampered IC13 dataset](https://github.com/wangyuxin87/Tampered-IC13), thanks for their great pioneering work!

---
### Citation
If you use the OSTF dataset, the Texture Jitter method or the DAF method, please cite this paper.
```
@inproceedings{ostf,
  title={Revisiting tampered scene text detection in the era of generative AI},
  author={Qu, Chenfan and Zhong, Yiwu and Guo, Fengjun and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={694--702},
  year={2025}
}
```

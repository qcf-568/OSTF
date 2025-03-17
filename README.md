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
2. Download the dataset ***training set images*** from [ArT](https://rrc.cvc.uab.es/?ch=14&com=introduction), [ICDAR2013](https://rrc.cvc.uab.es/?ch=2&com=introduction), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction), [ICDAR2017-MLT](https://rrc.cvc.uab.es/?ch=8&com=introduction), [LSVT](https://rrc.cvc.uab.es/?ch=16&com=introduction), [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=introduction), [TextOCR](https://textvqa.org/textocr/dataset/).
3. Rename the 7 downloaded image dirs into an "img" dir under the 7 sub-dirs. For example, "mv [Your downloaded ArT train images] pretrain/ArT/img" and "mv [Your downloaded ReCTS train images] pretrain/ReCTS/img".
4. Make a new dir named "revjpegs" in this main dir, and make sub-dirs to make sure that the dir "revjpegs" has the same sub-dir structure as the "pretrain" dir. For example, it should has the dirs "revjpegs/ArT/img" and "revjpegs/ReCTS/img", etc, corresponding to "pretrain/ArT/img" and "pretrain/ReCTS/img" respectively.
5. Download the fbcnn_color.pth following this [Readme.md](https://github.com/qcf-568/OSTF/tree/main/FBCNN#readme). In the [FBCNN dir](https://github.com/qcf-568/OSTF/tree/main/FBCNN), run the command to create reverse jpeg images for each of the 7 sub-dir images of the pretrain dir. For example, run "CUDA_VISIBLE_DEVICES=0 python app.py --inp pretrain/ArT/img/ --out revjpegs/ArT/img/" and "CUDA_VISIBLE_DEVICES=0 python app.py --inp pretrain/ReCTS/img/ --out revjpegs/ReCTS/img/".

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
revjpeg---ArT---img---....
  |         |     |
  |         |   train.pk
  |         |
  |        ICDAR2015---img---...
  |         |           |
  |         |         train.pk
  |         |
  |        ...
  |
mmcv_custom---...
  |
mmdet---...
  |
tools---...
  |
mmacc---...
```
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

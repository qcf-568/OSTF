# Revisiting Tampered Scene Text Detection in the Era of Generative AI [AAAI2025]

### This is the official implementation of the paper Revisiting Tampered Scene Text Detection in the Era of Generative AI.  [paper](https://arxiv.org/pdf/2407.21422)

---

### The Open-Set Text Forensics (OSTF) dataset is now publicly available at [Google Drive](https://drive.google.com/file/d/16Pyv7nLBOsOefwzdCsa0ndXxnzknfxtw/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/10FbI3SfWWV92vqv3X-ILxQ?pwd=OSTF). 

<font size=10>Researchers are welcome 😃 to apply for this dataset by sending an email to  202221012612@mail.scut.edu.cn (with institution email address) and introducing:</font><br/>
1. Who you are and your institution.
2. Who is your supervisor/mentor.
---
### Texture Jitter train data preparation
1. Download and unzip the [pretrain_pk.zip](https://drive.google.com/file/d/1xvu82bZvgq7TBXEjByFvuGi6th5ifsHY/view?usp=sharing) in this dir. After unzip, you will get a new dir named "pretrain" with 7 sub-dirs (ArT, ICDAR2013, ICDAR2015, ICDAR2017-MLT, LSVT, ReCTS, TextOCR).
2. Download the dataset images from [ArT](https://rrc.cvc.uab.es/?ch=14&com=introduction), [ICDAR2013](https://rrc.cvc.uab.es/?ch=2&com=introduction), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction), [ICDAR2017-MLT](https://rrc.cvc.uab.es/?ch=8&com=introduction), [LSVT](https://rrc.cvc.uab.es/?ch=16&com=introduction), [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=introduction), [TextOCR](https://textvqa.org/textocr/dataset/).
3. Rename the 7 downloaded image dirs into an "img" dir under the 7 sub-dirs. For example, "mv [Your downloaded ArT train images] pretrain/ArT/img" and "mv [Your downloaded ReCTS train images] pretrain/ReCTS/img".
4. Make a new dir named "revjpegs" in this main dir, and make sub-dirs to make sure that the dir "revjpegs" has the same sub-dir structure as the "pretrain" dir. For example, it should has the dirs "revjpegs/ArT/img" and "revjpegs/ReCTS/img", etc.
5. Download the fbcnn_color.pth . Run the command to create revjpeg images for each of the 7 sub-dir images of the pretrain dir.
---
### Train

Enviroment based on Python 3.9.12

pip install -r requirements.txt

---

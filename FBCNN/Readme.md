### FBCNN
 
This dir is used to generated the "reverse jpeg compressed" images used in the Texture Jitter paradigm, and is modified from [the official repo](https://github.com/jiaxi-jiang/FBCNN).

#### Usage
1. Download the model fbcnn_color.pth from [Google Drive]() into this dir.
2. run "CUDA_VISIBLE_DEVICES=0 python app.py --inp [your input images dir] --out [the output images dir]" to reverse compress your images.

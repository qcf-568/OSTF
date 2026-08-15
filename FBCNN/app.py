# import gradio as gr
import os.path
import numpy as np
from collections import OrderedDict
import torch
import cv2
from PIL import Image, ImageOps
import utils_image as util
from network_fbcnn import FBCNN as net
import requests
import datetime
import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--inp', type=str, required=True, help='input_dir')
parser.add_argument('--out', type=str, required=True, help='output_dir')
args = parser.parse_args()

def inference(model, input_img, is_gray, input_quality, zoom=100, x_shift=0, y_shift=0):
    
    # print("datetime:",datetime.datetime.utcnow())
    input_img_width, input_img_height = Image.fromarray(input_img).size
    # print("img size:",(input_img_width,input_img_height))
    
    # if (input_img_width > 1080) or (input_img_height > 1080):
    #     resize_ratio = min(1080/input_img_width, 1080/input_img_height)
    #     resized_input = Image.fromarray(input_img).resize((int(input_img_width*resize_ratio)+(input_img_width*resize_ratio < 1),
    #                                                        int(input_img_height*resize_ratio)+(input_img_height*resize_ratio < 1)),
    #                                                       resample=Image.BICUBIC)
    #     input_img = np.array(resized_input)
    #     print("input image resized to:", resized_input.size)

    if is_gray:
        n_channels = 1 # set 1 for grayscale image, set 3 for color image
        model_name = 'fbcnn_gray.pth'
    else:
        n_channels = 3 # set 1 for grayscale image, set 3 for color image
        model_name = 'fbcnn_color.pth'
    # nc = [64,128,256,512]
    # nb = 4
    

    input_quality = 100 - input_quality

    # model_path = model_name

    # if os.path.exists(model_path):
    #     print(f'{model_path} already exists.')
    # else:
    #     print("downloading model")
    #     os.makedirs(os.path.dirname(model_path), exist_ok=True)
    #     url = 'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{}'.format(os.path.basename(model_path))
    #     r = requests.get(url, allow_redirects=True)
    #     open(model_path, 'wb').write(r.content)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("device:",device)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnrb'] = []

    # ------------------------------------
    # (1) img_L
    # ------------------------------------

    # print("#if n_channels")
    if n_channels == 1:
        open_cv_image = Image.fromarray(input_img)
        open_cv_image = ImageOps.grayscale(open_cv_image)
        open_cv_image = np.array(open_cv_image) # PIL to open cv image
        img = np.expand_dims(open_cv_image, axis=2)  # HxWx1
    elif n_channels == 3:
        open_cv_image = np.array(input_img) # PIL to open cv image
        if open_cv_image.ndim == 2:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)  # RGB

    # print("#util.uint2tensor4(open_cv_image)")
    img_L = util.uint2tensor4(open_cv_image)
    
    # print("#img_L.to(device)")
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------
    
    # print("#model(img_L)")
    img_E,QF = model(img_L)
    # print("#util.tensor2single(img_E)")
    img_E = util.tensor2single(img_E)
    # print("#util.single2uint(img_E)")
    img_E = util.single2uint(img_E)
    
    # print("#torch.tensor([[1-input_quality/100]]).cuda() || torch.tensor([[1-input_quality/100]])")
    qf_input = torch.tensor([[1-input_quality/100]]).cuda() if device == torch.device('cuda') else torch.tensor([[1-input_quality/100]])
    # print("#util.single2uint(img_E)")
    img_E,QF = model(img_L, qf_input)  

    # print("#util.tensor2single(img_E)")
    img_E = util.tensor2single(img_E)
    # print("#util.single2uint(img_E)")
    img_E = util.single2uint(img_E)

    if img_E.ndim == 3:
        img_E = img_E[:, :, [2, 1, 0]]
    
    # print("--inference finished")

    out_img = Image.fromarray(img_E)
    out_img_w, out_img_h = out_img.size # output image size
    zoom = zoom/100
    x_shift = x_shift/100
    y_shift = y_shift/100
    zoom_w, zoom_h = out_img_w*zoom, out_img_h*zoom
    zoom_left, zoom_right = int((out_img_w - zoom_w)*x_shift), int(zoom_w + (out_img_w - zoom_w)*x_shift)
    zoom_top, zoom_bottom = int((out_img_h - zoom_h)*y_shift), int(zoom_h + (out_img_h - zoom_h)*y_shift)
    in_img = Image.fromarray(input_img)
    in_img = in_img.crop((zoom_left, zoom_top, zoom_right, zoom_bottom))
    in_img = in_img.resize((int(zoom_w/zoom), int(zoom_h/zoom)), Image.CUBIC)
    out_img = out_img.crop((zoom_left, zoom_top, zoom_right, zoom_bottom))
    out_img = out_img.resize((int(zoom_w/zoom), int(zoom_h/zoom)), Image.CUBIC)
    
    # print("--generating preview finished")
    
    return img_E, in_img, out_img

# print(f'loading model from {model_path}')
model = net(in_nc=3, out_nc=3, nc=[64,128,256,512], nb=4, act_mode='R')
# print("#model.load_state_dict(torch.load(model_path), strict=True)")
model.load_state_dict(torch.load('fbcnn_color.pth'), strict=True)
print("#model.eval()")
model.eval()
print("#for k, v in model.named_parameters()")
for k, v in model.named_parameters():
    v.requires_grad = False
print("#model.to(device)")
model = model.cuda()
print("Model loaded.")
if not os.path.exists(args.out):
    os.mkdir(args.out)
imgp=args.inp
for i in tqdm(range(len(os.listdir(args.inp)))):
    if os.path.exists(imgp+'/%d.jpg'%i):
        try:
            inference(model,cv2.cvtColor(cv2.imread(imgp+'/%d.jpg'%i),cv2.COLOR_BGR2RGB),False,100)[2].save(args.out+'/%d.jpg'%i,quality=100)
        except:
            print('error')
            continue
# gr.Interface(
#     fn = inference,
#     inputs = [gr.inputs.Image(label="Input Image"),
#               gr.inputs.Checkbox(label="Grayscale (Check this if your image is grayscale)"),
#               gr.inputs.Slider(minimum=1, maximum=100, step=1, label="Intensity (Higher = stronger JPEG artifact removal)"),
#               gr.inputs.Slider(minimum=10, maximum=100, step=1, default=50, label="Zoom Image "
#                                                                                   "(Use this to see a copy of the output image up close. "
#                                                                                    "100 = original size)"),
#               gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Zoom horizontal shift "
#                                                                      "(Increase to shift to the right)"),
#               gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Zoom vertical shift "
#                                                                      "(Increase to shift downwards)")
#               ],
#     outputs = [gr.outputs.Image(label="Result"),
#                gr.outputs.Image(label="Before:"),
#                gr.outputs.Image(label="After:")],
#     examples = [["doraemon.jpg",False,60,42,50,50],
#                ["tomandjerry.jpg",False,60,40,57,44],
#                ["somepanda.jpg",True,100,30,8,24],
#                ["cemetry.jpg",False,70,20,76,62],
#                ["michelangelo_david.jpg",True,30,12,53,27],
#                ["elon_musk.jpg",False,45,15,33,30],
#                ["text.jpg",True,70,50,11,29]],
#     title = "JPEG Artifacts Removal [FBCNN]",
#     description = "Gradio Demo for JPEG Artifacts Removal. To use it, simply upload your image, "
#                   "or click one of the examples to load them. Check out the paper and the original GitHub repo at the links below. "
#                   "JPEG artifacts are noticeable distortions of images caused by JPEG lossy compression. "
#                   "This is not a super-resolution AI but a JPEG compression artifact remover. "
#                   "Written below the examples are the limitations of the input image. ",
#     article = "<p style='text-align: left;'>Uploaded images with a length longer than 1080 pixels will be downscaled to a smaller size "
#               "with a length of 1080 pixels. Uploaded images with transparency will be incorrectly reconstructed at the output.</p>"
#               "<p style='text-align: center;'><a href='https://github.com/jiaxi-jiang/FBCNN'>FBCNN GitHub Repo</a><br>"
#               "<a href='https://arxiv.org/abs/2109.14573'>Towards Flexible Blind JPEG Artifacts Removal (FBCNN, ICCV 2021)</a><br>"
#               "<a href='https://jiaxi-jiang.github.io/'>Jiaxi Jiang, </a>"
#               "<a href='https://cszn.github.io/'>Kai Zhang, </a>"
#               "<a href='http://people.ee.ethz.ch/~timofter/'>Radu Timofte</a></p>",
#     allow_flagging="never"
# ).launch(enable_queue=True)

import argparse

# create an argument parser
parser = argparse.ArgumentParser(description='Image morphing program')
# define the command-line arguments
parser.add_argument('--face', type=str, help='path to the face image file')
parser.add_argument('--body', type=str, help='path to the body image file')
# parse the arguments
args = parser.parse_args()

# check if the face and body image paths are provided
if args.face is None or args.body is None:
    raise ValueError('Please provide paths to the face and body images using the --face and --body options')

import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
import cv2
import torch
from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import get_target
from utils.inference.core import model_inference
from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions


# --- Initialize models ---
app = Face_detect_crop(name="antelope", root="./insightface_func/models")
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

# main model for generation
G = AEI_Net(backbone="unet", num_blocks=2, c_id=512)
G.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G.load_state_dict(torch.load("weights/G_unet_2blocks.pth", map_location=torch.device(device)))
G = G.cuda()
G = G.half()

# arcface model to get face embedding
netArc = iresnet100(fp16=False)
netArc.load_state_dict(torch.load("arcface_model/backbone.pth"))
netArc = netArc.cuda()
netArc.eval()

# model to get face landmarks
handler = Handler("./coordinate_reg/model/2d106det", 0, ctx_id=0, det_size=640)

# model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
use_sr = True
if use_sr:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    opt = TestOptions()
    # opt.which_epoch ='10_7'
    model = Pix2PixModel(opt)
    model.netG.train()

# use the provided image paths
face_path = args.face
body_path = args.body

face_full = cv2.imread(face_path)
crop_size = 224  # don't change this
batch_size = 40

face = crop_face(face_full, app, crop_size)[0]
face = [face[:, :, ::-1]]

body_full = cv2.imread(body_path)
full_frames = [body_full]
body = get_target(full_frames, app, crop_size)

final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
    full_frames, face, body, netArc, G, app, set_target=False, crop_size=crop_size, BS=batch_size
)

result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
cv2.imwrite("result.png", result)

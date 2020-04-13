import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

import numpy as np
from PIL import Image
import cv2, pdb, glob, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvison.models as models

import pdb




parser = argparse.ArgumentParser(description='Deeplab Segmentation')
parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to save the output results. (required)')
args=parser.parse_args()

dir_name=args.input_dir;

download_path = 'pretrained/deeplabv3_resnet50_coco-cd0a2569.pth'


## setup ####################

LABEL_NAMES = np.asarray([
	'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
	'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
	'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


MODEL = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21, aux_loss=None)

MODEL = DeepLabModel(download_path)

state_dict = torch.load(download_path)
MODEL.load_state_dict(state_dict)

MODEL.eval()

print('model loaded successfully!')


normalize = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


#######################################################################################


list_im=glob.glob(dir_name + '/*_img.png'); list_im.sort()


for i in range(0,len(list_im)):

	image = Image.open(list_im[i]).convert("RGB")
        tensor_image = normalize(image).unsqueeze(0)

	result = MODEL(tensor_image)

        result = result.data.cpu().numpy()

        pdb.set_trace()

	#seg=cv2.resize(seg.astype(np.uint8),image.size)

	#mask_sel=(seg==15).astype(np.float32)


	#name=list_im[i].replace('img','masksDL')
	#cv2.imwrite(name,(255*mask_sel).astype(np.uint8))

str_msg='\nDone: ' + dir_name
print(str_msg)



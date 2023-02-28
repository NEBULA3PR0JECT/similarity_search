import sys
import os
import numpy as np
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
############################# Load BLIP Model ####################################################################
def load_image(image_size, img_url, device):
  
    raw_image = Image.open(img_url).convert('RGB')
    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

############################# Image Captioning ####################################################################
sys.path.append('/notebooks/BLIP')
from PIL import Image
import glob
from models.blip import blip_decoder

image_size = 384

image_dic = {}
image_list = sorted(glob.glob('/notebooks/images/*.jpg'))
for img_url in image_list :
    image_name = img_url.split('/')[-1]
    image = load_image(image_size=image_size, img_url=img_url ,device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        image_dic[image_name] = caption[0]
        
############################# Image-Text Matching ##################################################################
from models.blip_itm import blip_itm

caption_list = image_dic.values()
image_size = 384

#####Define distance matrix
dist = np.zeros((len(caption_list),len(caption_list)))

for img_idx, img_url in enumerate(image_list):
    image = load_image(image_size=image_size, img_url=img_url ,device=device)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
        
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    for cap_idx ,cap in enumerate(caption_list):
        itm_output = model(image,cap,match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        #print('The image and text is matched with a probability of %.4f'%itm_score)
        itc_score = model(image,cap,match_head='itc')
        #print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
        dist[img_idx,cap_idx] = itc_score


print(dist)





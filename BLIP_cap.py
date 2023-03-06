import sys
sys.path.append('/notebooks/BLIP')
import os
import glob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sbn

import matplotlib as mpl

import random 
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from models.blip_itm import blip_itm



img_url_example = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
image_size = 384
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
def cap_image(image_size=image_size,img_url=img_url_example, device=device):
    
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
    return image_name,caption[0]
      
############################# Image Caption Dictionary ##############################################################
image_dic = {}
image_list = sorted(glob.glob('/notebooks/images/*.jpg'))

for img_url in image_list:
    image_name, caption = cap_image(img_url=img_url)
    image_dic[image_name] = caption

print ("cap model done")  

caption_list = list(image_dic.values())
############################# Image-Text Matching ##################################################################
def matching_image_cap(number_cap=len(caption_list), image_list=image_list, caption_list=caption_list, image_size=image_size, device=device):
    #del caption_list
    caption_samp = random.sample(caption_list, k=number_cap)
    #caption_list = order_samp_list(caption_list,caption_samp)
    
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
            itc_score = model(image,cap,match_head='itc')*100
            #print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
            #itc_score = torch.round(itc_score, decimals = 0)
            itc_score = itc_score.type(torch.int64)
            dist[img_idx,cap_idx] = itc_score
    return dist, caption_list

######################## Order Sample List ######################################
def order_samp_list(caption_list,caption_samp):
    
    order_list = np.full_like(caption_list, np.nan, dtype=np.double).astype(str)
    
    for smp in caption_samp:
        inx = caption_list.index(smp)
        order_list[inx] = smp
    
    order_list = order_list[order_list != 'nan']
    
    return order_list

######################## Order Image-Text Matching  ######################################
def order_samp_list(caption_list, distance_map):
    
    match_cap_inx = np.argmax(distance_map, axis = 1)
    
    for i in range(len(caption_list)):
        caption_list[i] = caption_list[match_cap_inx[i]]
    
    return caption_list

############### heat map for validation #########################
k = 10
dist = np.zeros((len(caption_list),k))

distance_map = matching_image_cap(k, image_list, caption_list)[0]
print(distance_map)

fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(distance_map)
ax.set_title("distances", size=20)
fig.tight_layout()
plt.savefig("distance_heatmap.png",format='png',dpi=150)


########################### Change to Matchest Caption ########## 
# caption_list_new = order_samp_list(caption_list, distance_map)
# distance_map = matching_image_cap(k, image_list, caption_list_new)[0]
# print(distance_map)
# fig, ax = plt.subplots(figsize=(10,10))
# im = ax.imshow(distance_map)
# ax.set_title("distances", size=20)
# fig.tight_layout()
# plt.savefig("distance_heatmap_correct.png",format='png',dpi=150)
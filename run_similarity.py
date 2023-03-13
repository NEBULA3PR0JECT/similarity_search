import BLIP_cap as blip
import similarity as sim

import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib as mpl

image_size = 384
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################# Image Caption Dictionary ###############

image_dic = {}
image_list = sorted(glob.glob('/notebooks/images/alison/*.jpg'))

for img_url in image_list:
    image_name, caption = blip.cap_image(img_url=img_url)
    image_dic[image_name] = caption

print ("cap model done")  

caption_list = list(image_dic.values())

############### Distance Map #########################################

k = 7
distance_map = blip.matching_image_cap(k, image_list, caption_list)[0]
print(distance_map)

############### heat map for validation ##############################

fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(distance_map)
ax.set_title("distances", size=20)
fig.tight_layout()
plt.savefig("distance_heatmap_alison.png",format='png',dpi=150)


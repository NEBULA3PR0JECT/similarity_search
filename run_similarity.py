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
plt.savefig("distance_heatmap_alison_neg.png",format='png',dpi=150)

cost , cost_pre_flip = sim.comp_accum_cost_matrix(distance_map)
print(cost_pre_flip)
# print(cost)

path, path_idx, sum_path = sim.warp_path(cost)
print(path, path_idx, sum_path)



# fig, ax = plt.subplots(figsize=(12, 8))
# ax = sbn.heatmap(cost_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
# ax.invert_yaxis()


# # Get the warp path in x and y directions
# path_x = [p[0] for p in path_idx]
# path_y = [p[1] for p in path_idx]

# # Align the path from the center of each cell
# path_xx = [x+0.5 for x in path_x]
# path_yy = [y+0.5 for y in path_y]

# ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)

# fig.savefig("alison_heatmap_path.png", **savefig_options)



import BLIP_cap as blip
import similarity as sim

import torch


image_size = 384
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


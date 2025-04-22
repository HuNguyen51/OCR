d_model = 256
num_heads = 8
d_ff = 2048
num_decoder_layers = 6
dropout = 0.1

maxlen = 256

image_height=32
image_min_width=32
image_max_width=512

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

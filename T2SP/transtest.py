import torch
import numpy as np
from core.T2SP import T2SP

model = T2SP(embed_dim = 512, n_head = 8, n_layer = 6, res = 512, num_nodes = 10)

test = torch.rand((10,2))

logits = model(test)
print(logits, logits.shape)


import torch, math
import numpy as np
import torch.nn as nn

class CartesianEmbedding(nn.Module):
    
  def __init__(self, embed_dim, res):
    super(CartesianEmbedding, self).__init__()

    self.embed_dim = embed_dim
    self.res = res

    self.embed = nn.Embedding(res, embed_dim)

  def indexify(self, nodes):
    return torch.LongTensor([[math.floor(n[0]*self.res), math.floor(n[1]*self.res)] for n in nodes])

  def forward(self, x):
    return self.embed(self.indexify(x)).reshape(-1,2*self.embed_dim)


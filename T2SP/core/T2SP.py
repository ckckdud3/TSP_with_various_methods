import torch
import numpy as np
import torch.nn as nn

from core.CartesianEmbedding import CartesianEmbedding

class T2SP(nn.Module):
    
  def __init__(self, embed_dim, n_head, n_layer, res, num_nodes):

    super(T2SP,self).__init__()

    assert embed_dim % 2 == 0, 'embed_dim must be even number'

    self.embed_dim = embed_dim
    self.n_head = n_head
    self.n_layer = n_layer
    self.res = res
    self.num_nodes = num_nodes

    self.emb = CartesianEmbedding(embed_dim=int(embed_dim/2), res=res)
    self.encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead=n_head)
    self.encoder = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers=n_layer)
    self.fc = nn.Linear(embed_dim, num_nodes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):

    embed_x = self.emb(x)
    enc_x = self.encoder(embed_x)
    logits = self.fc(enc_x)
    logits = self.softmax(logits)

    return logits
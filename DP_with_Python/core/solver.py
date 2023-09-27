import numpy as np
from .utils import samplereader, case


class solver:
    
  def __init__(self, num_nodes, file_name):
        
    self.num_nodes = num_nodes
    self.file_name = file_name

    self.reader = samplereader(self.num_nodes, self.file_name)

  def __str__(self):
    return f'[TSP Solver with DP] Number of Nodes : {self.num_nodes}, Input file name : {self.file_name}'
  def convert_to_adm(self, data: case):
        
    points = np.array(data.nodes)
    adm = np.zeros((self.num_nodes, self.num_nodes))
    
    for i in range(self.num_nodes):
      for j in range(i+1, self.num_nodes):
        dist = ((points[i] - points[j])**2).sum()**0.5
        adm[i][j] = dist
        adm[j][i] = dist
    
    return adm
        
  def process(self):
        
    a = self.reader.read_one()

        
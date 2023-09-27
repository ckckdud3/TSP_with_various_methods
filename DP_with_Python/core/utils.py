import logging

from dataclasses import dataclass
from typing import List

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s : %(message)s')

@dataclass
class case:
  nodes: List[List[float]]
  answer: List[int]


class samplereader:

  def __init__(self, num_nodes: int, file_name: str):
    
    self.num_nodes = num_nodes
    self.file_name = file_name
    self.f = None
    try:
      self.f = open(self.file_name, 'r')
    except:
      raise ValueError('Invalid file name')

  def read_one(self):
    
    if self.f.closed:
      logger.info('File closed')
      return None
    
    line = self.f.readline().split(' ')

    if not line:
      self.f.close()
      logger.info('EOF reached')
      return None
    
    nodes = []
    answer = []

    for i in range(self.num_nodes):
      nodes.append([float(line[i*2]), float(line[i*2+1])])

    answer = [int(i) for i in line[-self.num_nodes-1:-1]]

    ret = case(nodes=nodes, answer = answer)

    return ret
    

    
import numpy as np
from .utils import samplereader, case

import logging

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s : %(message)s')

INF = 1e10

class solver:
    
  def __init__(self, num_nodes, file_name):
        
    self.num_nodes = num_nodes
    self.file_name = file_name

    self.adm = None
    self.memo = None
    self.answer = None

    self.reader = samplereader(self.num_nodes, self.file_name)

  def __str__(self):
    return f'[TSP Solver with DP] Number of Nodes : {self.num_nodes}, Input file name : {self.file_name}'
  
  def __convert_to_adm(self, data: case):
        
    points = np.array(data.nodes)
    adm = np.zeros((self.num_nodes, self.num_nodes))
    
    for i in range(self.num_nodes):
      for j in range(i+1, self.num_nodes):
        dist = ((points[i] - points[j])**2).sum()**0.5
        adm[i][j] = dist
        adm[j][i] = dist
    
    return adm

  def __process_internal(self, pos, visited):

    if visited == (1 << self.num_nodes) - 1 :
      self.memo[pos][visited] = self.adm[pos][0]
      return self.adm[pos][0]
    
    if self.memo[pos][visited] != INF:
      return self.memo[pos][visited]
    
    for i in range(1,self.num_nodes):
      if visited & (1 << i):
        continue
      self.memo[pos][visited] = min(self.memo[pos][visited], self.__process_internal(i, visited | (1 << i)) + self.adm[pos][i])

    return self.memo[pos][visited]
  
  def __append_path(self, pos, visited):

    if pos == 0:
      self.answer[0].append(pos + 1)
    else:
      self.answer[0].append(pos + 1)

    next = [INF, 0]
    for i in range(self.num_nodes):
      if visited & (1 << i):
        continue
      if (self.adm[pos][i] + self.memo[i][visited | (1 << i)]) < next[0]:
        next[0] = self.adm[pos][i] + self.memo[i][visited | (1 << i)]
        next[1] = i

    self.__append_path(next[1], visited | (1 << next[1]))

  def process(self):
    
    while True:
      data = self.reader.read_one()

      if data is not None:
        self.adm = self.__convert_to_adm(data=data)
        self.memo = [[INF] * (1 << self.num_nodes) for _ in range(self.num_nodes)]
        self.answer = [[], 0.0]

        self.answer[1] = self.__process_internal(0, 1)
        self.__append_path(0, 1)
        self.answer += ' 1'

      else:
        break

    logger.info(f'Processing of file \'{self.file_name}\' finished')
    return
    



        
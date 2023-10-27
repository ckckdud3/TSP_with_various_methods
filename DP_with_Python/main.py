import argparse
from core.DPsolver import DPsolver

parser = argparse.ArgumentParser(description='Concorde TSP solver with Dynamic Programming')

parser.add_argument('filename', type=str, help='Testcase file path.')
parser.add_argument('num_nodes', type=int, help='Number of nodes per graph.')

args = parser.parse_args()

a = DPsolver(args.num_nodes, args.filename)
a.process()
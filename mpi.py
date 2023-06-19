import numpy as np

import mpi4py 
mpi4py.rc.threads = False 
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("Hello from rank", rank)
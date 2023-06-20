import numpy as np

import mpi4py 
mpi4py.rc.threads = False 
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Hello from rank", rank)

if rank == 0:
    A = np.arange(10)
    print("Rank", rank, "has A =", A)
    A_split = np.array_split(A, size)
    print("Rank", rank, "has A_split =", A_split)
else:
    A = None
    A_split = None
    print("Rank", rank, "has A =", A)

# Send the appopiate A_split (list of ndarrays) to the owner ranks
print("Rank", rank, "is scattering")
A = comm.scatter(A_split, root=0)
print("Rank", rank, "has A =", A, "type =", type(A))

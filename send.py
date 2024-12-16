import numpy as np
import mpi4py
from mpi4py import MPI

# MPI variables for the communications inter-processes
comm = MPI.COMM_WORLD
NbP = comm.Get_size() # Nb of processes
Me = comm.Get_rank() # Rank of actual process

# Local variables of the process for its computations
n = 4096
V = np.empty(n,np.float64)
M = np.empty([n,n],np.float64) 
Vout = np.zeros(n,np.float64)
Norm = np.zeros(1,np.float64)
Res = np.zeros(1,np.float64)

# Init of local V and M arrays from data files

# Computation-Communication LOOP:
for i in range (NbP):
    # Local computations
    Vout = np.matmul(M,V) # Matrix x Vector multiplication
    Norm[0] = np.linalg.norm(Vout) # Norm
    Res[0] += Norm[0] # Sum of norms

    # Data circulation (of V array)
    dest = (Me + 1) % NbP 
    source = (Me - 1 + NbP) % NbP
    comm.Sendrecv_replace(V, dest=dest, source=source)

if Me == 0:
    print("Final result: ", Res[0])

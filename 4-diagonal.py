import numpy as np
import mpi4py
from mpi4py import MPI

# MPI variables for the communications inter-processes
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

# Local variables of the process for its computations
n = 4096
V = np.empty([n,n],np.float64)
q = 0.0

# STEP 0: Init of V array from data files
V.fill(1.0)

# STEP 1: Computation
for i in range(n):
    for j in range(n):
        V[i][j] = i + j

# STEP 2: Communication TO DO
# Each process sends its diagonal elements (V[i][i], 0 ≤ i < n) to its
# right neighbour in the ring and replace its diagonal with the elements
# received from its left neighbour
diagonal = np.empty(n, np.float64)
for i in range(n):
    diagonal[i] = V[i][i]

dest = (Me + 1) % NbP
source = (Me - 1 + NbP) % NbP

comm.Sendrecv_replace(diagonal, dest=dest, source=source)

for i in range(n):
    V[i][i] = diagonal[i]

# STEP 3: Computation
for i in range(n):
    for j in range(n):
        V[i][j] = i * j

# STEP 4:Communication & computation # r1 is the mean of V[0][0] on all processes 0, 1, 2, 3, 4, …
# and must be calculated over at least one process

r1 = comm.reduce(V[0][0], op=MPI.SUM, root=0)
r1 /= NbP

print(f"Process {Me}: Mean r1 = {r1}")
print(V)

# STEP 5: Communication & computation TO DO
# r2 is the sum of V[n-1][n-1] on processes with even number : 0, 2, …
# and must be calculated over at least one process

if Me % 2 == 0:
    local_value = V[n-1][n-1]
r2 = comm.reduce(local_value, op=MPI.SUM, root=0)

print(f"Process {Me}: Sum r2 = {r2}")

# STEP 6: Computation & communication # q must be computed as follows: q = r1/r2
# and finally q must be available on each process
q = 0.0
if Me == 0:
    if r2 != 0:  # Evitar divisão por zero
        q = r1 / r2
    print(f"Processo {Me}: Valor final de q = {q}")

q = comm.bcast(q, root=0)

# STEP 7: Computation
# update of V array on each process:
V = q*V
# STEP 8: Save V array on disk


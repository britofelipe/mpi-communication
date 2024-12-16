import numpy as np
import mpi4py
from mpi4py import MPI

# MPI variables for the communications inter-processes
comm = MPI.COMM_WORLD
P = comm.Get_size()
Me = comm.Get_rank()

# Local variables of the process for its computations
N = 64 # total number of data in the Grid
totalSide = np.sqrt(N) # we assume 𝑵 is a square integer
gridSide = np.sqrt(P) # we assume 𝑷 is a square integer
localSide = int(totalSide / gridSide) # we assume √𝑵 is a multiple of √𝑷

# Local 2D-array of the process
localTab = np.zeros((localSide, localSide), dtype=np.float64)

# Buffer storing the south frontier of the upper process
recvNorthFrontier = np.zeros(localSide, dtype=np.float64)
recvSouthFrontier = np.zeros(localSide, dtype=np.float64)
recvEastFrontier = np.zeros(localSide, dtype=np.float64)
recvWestFrontier = np.zeros(localSide, dtype=np.float64)

#number of iterations
NbIter = 10000

def RankToCoord(rank,gridSide): 
    return divmod(rank,gridSide)
#rank  coord conversion function

def CoordToRank(line,col,gridSide): #coord  rank conversion function
    return(line*gridSide+col)

line, col = RankToCoord(Me, gridSide)

north = CoordToRank((line - 1) % gridSide, col, gridSide)
south = CoordToRank((line + 1) % gridSide, col, gridSide)
west = CoordToRank(line, (col - 1) % gridSide, gridSide)
east = CoordToRank(line, (col + 1) % gridSide, gridSide)

# computation-communication loop --------------------------------------
for step in range(NbIter):
    comm.Sendrecv(np.ascontiguousarray(localTab[1, :]), dest=north, sendtag=0,
                  recvbuf=recvNorthFrontier, source=north, recvtag=0)
    comm.Sendrecv(np.ascontiguousarray(localTab[-2, :]), dest=south, sendtag=1,
                  recvbuf=recvSouthFrontier, source=south, recvtag=1)

    comm.Sendrecv(np.ascontiguousarray(localTab[:, -2]), dest=east, sendtag=2,
                  recvbuf=recvEastFrontier, source=east, recvtag=2)
    comm.Sendrecv(np.ascontiguousarray(localTab[:, 1]), dest=west, sendtag=3,
                  recvbuf=recvWestFrontier, source=west, recvtag=3)

    localTab[0, :] = recvNorthFrontier
    localTab[-1, :] = recvSouthFrontier
    localTab[:, 0] = recvWestFrontier
    localTab[:, -1] = recvEastFrontier
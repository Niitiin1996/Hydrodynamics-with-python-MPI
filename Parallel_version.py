import numpy as np
import copy
import math
from mpi4py import MPI

def global_index(x):
  return int(rank * (N/size) + x)

def communication_above(theone,above):
    if  rank < size-1:
        comm.Send(theone, dest = rank+1, tag=13)

    if rank > 0 and rank < size:
        comm.Recv(above, source=rank-1, tag=13)

        
def communication_below(theone,below):
    if  rank > 0 and rank < size:
        comm.Send(theone, dest = rank-1, tag=14)

    if rank < size-1:
        comm.Recv(below, source=rank+1, tag=14)

def divergence(ghost_below,vel_local,ghost_above):
  for i in range(local_size):
    if global_index(i) > 0 and global_index(i) < N-1:
            for j in range(1,N-1):
              if i+1 > local_size-1 :
                div_vel_local[i][j] = (ghost_below[0][j] - vel_local[0][i-1,j] + vel_local[1][i,j+1] - vel_local[1][i,j-1])/(2*h)
              elif i-1 < 0:
                div_vel_local[i][j] = (vel_local[0][i+1,j] - ghost_above[0][j] + vel_local[1][i,j+1] - vel_local[1][i,j-1])/(2*h)
              else:
                div_vel_local[i][j] = (vel_local[0][i+1,j] - vel_local[0][i-1,j] + vel_local[1][i,j+1] - vel_local[1][i,j-1])/(2*h)
  return div_vel_local

#basics for parallelisations 
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
status = MPI.Status()

L=20
n=127
N=n+1
local_size = int(N/size)
h=L/n


#Velocity definition 
vel_local=np.array(((np.zeros((local_size,N))),(np.zeros((local_size,N)))))

x0 = -10
y0 = -10

for i in range(local_size):
            for j in range(0,N):
                vel_local[0][i,j] = (y0+h*j)/((x0+h*global_index(i))**2+(y0+h*j)**2+1)+ 2*(x0+h*global_index(i)) * math.exp(-(x0+h*global_index(i))**2-(y0+h*j)**2)
                vel_local[1][i,j] = (-x0-h*global_index(i))/((x0+h*global_index(i))**2+(y0+h*j)**2+1) + 2*(y0+h*j) * math.exp(-(x0+h*global_index(i))**2-(y0+h*j)**2)

comm.Barrier()
start = MPI.Wtime()  
#Calculation of divergence of the initial velocity
ghost_below = np.zeros((1,N))
ghost_above = np.zeros((1,N))
sum1=np.array(([0.0]))
sumall=np.array(([0.0]))

communication_below(vel_local[0][0,:],ghost_below)
communication_above(vel_local[0][local_size-1,:],ghost_above)

div_vel_local = np.zeros((local_size,N))

divergence(ghost_below,vel_local, ghost_above)
sum1[0] = np.linalg.norm(div_vel_local)**2
comm.Allreduce(sum1, sumall, op=MPI.SUM)

if rank == 0:
    print("Before = ",sumall**(1/2))


#Calculation of phi
phi_local = np.zeros((local_size,N))
phi_ghost_below = np.zeros((1,N))
phi_ghost_above = np.zeros((1,N))
phi_new_local = np.zeros((local_size,N))
c=1


tol = np.array(([1.0]))
while (1):
  sum=np.array(([0.0]))
  communication_below(phi_local[0][:],phi_ghost_below)
  communication_above(phi_local[local_size-1][:],phi_ghost_above)
  for i in range(local_size):
    if global_index(i) > 0 and global_index(i) < N-1:
      for j in range(1,N-1): 
       if i+1 > local_size-1:
          phi_new_local[i][j] = h*h/4*(div_vel_local[i][j]) + 1/4*(phi_ghost_below[0][j] + phi_local[i-1,j] + phi_local[i,j+1] + phi_local[i,j-1])
       elif i-1 < 0:
         phi_new_local[i][j] = h*h/4*(div_vel_local[i][j]) + 1/4*(phi_local[i+1,j] + phi_ghost_above[0][j] + phi_local[i,j+1] + phi_local[i,j-1])
       else:
         phi_new_local[i][j] = h*h/4*(div_vel_local[i][j]) + 1/4*(phi_local[i+1,j] + phi_local[i-1,j] + phi_local[i,j+1] + phi_local[i,j-1])
      sum[0] = sum[0] + ((phi_new_local[i][j]-phi_local[i][j])/(phi_local[i][j]+1e-40))**2
  
  c=c+1  
  comm.Allreduce(sum, tol, op=MPI.SUM)
  phi_local = copy.copy(phi_new_local)
  if tol**(0.5)<1e-40:
    break


#Calculation of gradiant

grad_phi_local = np.array(((np.zeros((local_size,N))),(np.zeros((local_size,N)))))

for i in range(local_size):
    if global_index(i) > 0 and global_index(i) < N-1:
      for j in range(1,N-1):
        if i+1 > local_size-1 :
          grad_phi_local[1][i,j]=(phi_new_local[i,j+1]-phi_new_local[i,j-1])/(2*h)
          grad_phi_local[0][i,j]=(phi_ghost_below[0][j]-phi_new_local[i-1,j])/(2*h)
        elif i-1 < 0 :
          grad_phi_local[1][i,j]=(phi_new_local[i,j+1]-phi_new_local[i,j-1])/(2*h)
          grad_phi_local[0][i,j]=(phi_new_local[i+1,j]-phi_ghost_above[0][j])/(2*h)
        else :
          grad_phi_local[1][i,j]=(phi_new_local[i,j+1]-phi_new_local[i,j-1])/(2*h)
          grad_phi_local[0][i,j]=(phi_new_local[i+1,j]-phi_new_local[i-1,j])/(2*h)


#Calculation of new velocity
vel_new_local = np.array(((np.zeros((local_size,N))),(np.zeros((local_size,N)))))
vel_new_local[0]=np.add(vel_local[0],grad_phi_local[0])
vel_new_local[1]=np.add(vel_local[1],grad_phi_local[1])

div_vel_new_local = np.zeros((local_size,N))
vel_new_ghost_below = np.zeros((1,N))
vel_new_ghost_above = np.zeros((1,N))

sum2=np.array(([0.0]))
sumall=np.array(([0.0]))

communication_below(vel_new_local[0][0,:],vel_new_ghost_below)
communication_above(vel_new_local[0][local_size-1,:],vel_new_ghost_above)

div_vel_new = np.zeros((N,N))
div_vel_new_local = divergence(vel_new_ghost_below,vel_new_local, vel_new_ghost_above)

sum2[0] = np.linalg.norm(div_vel_new_local)**2
comm.Allreduce(sum2, sumall, op=MPI.SUM)

comm.Barrier()
end = MPI.Wtime()  

if rank == 0:
    print("After = ", sumall**(1/2))
    print("The time =",end-start) 
    print("N=",N,"size=",size) 

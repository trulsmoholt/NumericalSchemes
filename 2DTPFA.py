import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from mpl_toolkits import mplot3d
import math

bottom_left = (0,0)
top_right = (1,1)


num_nodes_x = 12
num_nodes_y = 6
num_nodes = num_nodes_x*num_nodes_y
u_h = np.zeros(num_nodes)
nodes_x, nodes_y = np.meshgrid(np.linspace(bottom_left[0],top_right[0],num=num_nodes_x),np.linspace(bottom_left[1],top_right[1],num=num_nodes_y))
nodes = np.stack([nodes_x,nodes_y],axis=2)



x = sym.symbols('x')
y = sym.symbols('y')
u_fabric = x*y*(1-x)*(1-y)
f = 2*x*(1-x)+2*y*(1-y)

u_fabric_vec = np.zeros(num_nodes)
u_fabric_nodes = np.zeros((num_nodes_y,num_nodes_x))
f_vec = np.zeros(num_nodes)

A = np.zeros((num_nodes,num_nodes))
def meshToVec(j,i)->int:
    return i*num_nodes_y + j
def vecToMesh(h)->(int,int):
    return (h % num_nodes_y, math.floor(h/num_nodes_y))
def computeFlux(j,i):
    vec_i = meshToVec(j,i)

    north_i = meshToVec(j+1,i)
    south_i = meshToVec(j-1,i)
    east_i = meshToVec(j,i+1)
    west_i = meshToVec(j,i-1)
    north_face = (nodes[j,i+1,0]-nodes[j,i-1,0])/2
    south_face = north_face
    east_face = (nodes[j+1,i,1]-nodes[j-1,i,1])/2
    west_face = east_face

    dy = nodes[j+1,i,1]-nodes[j,i,1]
    A[vec_i,north_i] += -1*north_face*1/dy
    A[vec_i,vec_i] += 1*north_face*1/dy

    dx = nodes[j,i+1,0]-nodes[j,i,0]
    A[vec_i,east_i] += -1*east_face*1/dx
    A[vec_i,vec_i] += 1*east_face*1/dx

    dy = nodes[j,i,1]-nodes[j-1,i,1]
    A[vec_i,south_i] += -1*south_face*1/dy
    A[vec_i,vec_i] += 1*south_face*1/dy

    dx = nodes[j,i,0]-nodes[j,i-1,0]
    A[vec_i,west_i] += -1*west_face*1/dx
    A[vec_i,vec_i] += 1*west_face*1/dx
    return



def computeSource(j,i,x_d,y_d):
    north_face = (nodes[j,i+1,0]-nodes[j,i-1,0])/2
    south_face = north_face
    east_face = (nodes[j+1,i,1]-nodes[j-1,i,1])/2
    west_face = east_face
    f_vec[meshToVec(j,i)] = f.subs([(x,x_d),(y,y_d)])*north_face*east_face



for i in range(num_nodes_x):
    for j in range(num_nodes_y):
        vec_i = meshToVec(j,i)
        if (i==0) or (i==num_nodes_x-1) or (j==0) or (j==num_nodes_y-1):
            A[vec_i,vec_i] = 1
            f_vec[vec_i] = 0
            continue
        computeFlux(j,i)
        x_d = nodes[j,i,0]
        y_d = nodes[j,i,1]
        computeSource(j,i,x_d,y_d)
        u_fabric_vec[vec_i] = u_fabric.subs([(x,x_d),(y,y_d)])
        u_fabric_nodes[j,i] = u_fabric.subs([(x,x_d),(y,y_d)])
print(A)

#add boundaries


u_vec = np.linalg.solve(A,f_vec)
u_nodes = u_fabric_nodes.copy()
err_nodes = np.zeros(num_nodes)
err_nodes_max = 0
for i in range(num_nodes):
    u_nodes[vecToMesh(i)] = u_vec[i]
    err_nodes[i] = abs(u_fabric_vec[i]-u_vec[i])
err_nodes_max = np.max(err_nodes)
print('max error at nodes',err_nodes_max)



fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_surface(nodes[:,:,0],nodes[:,:,1],u_nodes,cmap='viridis', edgecolor='none')
ax.set_title('computed solution')
ax.set_zlim(0.00, 0.07)


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(nodes[:,:,0],nodes[:,:,1],u_fabric_nodes,cmap='viridis', edgecolor='none')
ax.set_title('exact solution')
ax.set_zlim(0.00, 0.07)





plt.show()


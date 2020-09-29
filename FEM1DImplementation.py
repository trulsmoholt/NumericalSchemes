
import numpy as np # Does all linear algebra
import sympy as sym # Symbolic library to make "lambda-like" functions
import matplotlib.pyplot as plt # Matplotlib imported for plotting
import math # Imported math to take square root
from scipy.interpolate import lagrange

#Define domain
x_0 = 0
x_n = 1
# Mesh
num_elements = 7
num_nodes = num_elements*2 + 1 #Number of nodes
nodes = np.linspace(x_0,x_n,num_nodes)
h = nodes[1]-nodes[0]
# Using sympy to make a fabricated solution to give source and boundary
x = sym.symbols('x')
u_fabric = x*(1-x)
u_fabric_nodes = np.zeros((num_nodes,1))
for i in range(num_nodes):
 u_fabric_nodes[i]=u_fabric.subs(x,nodes[i])
# Source
f = -u_fabric.diff(x,2)
# Boundary conditions
u_0 = u_fabric.subs(x,x_0)
u_prime_n = u_fabric.diff(x,1).subs(x,x_n)

#change method so that each element has 3 and not 2 nodes.
def ltgi(element, node):
    return 2*element+node

shape_function_0 = (1-2*x)*(1-x) 
shape_function_1 = 4*(1-x)*x
shape_function_2 = -(1-2*x)*x

shape_derivative_0 = shape_function_0.diff(x,1)
shape_derivative_1 = shape_function_1.diff(x,1)
shape_derivative_2 = shape_function_2.diff(x,1)

def global_to_local_map(node_0,node_1, x_g):
    return x_g*(node_1-node_0)+node_0
def jacobian_global_to_local_map(node_0,node_1):
    return node_1-node_0

A = np.zeros((num_nodes,num_nodes)) #Creating matrix of zeros
f_d = np.zeros((num_nodes,1)) #creating source vector of zeros
#Assembly
for element in range(num_elements):
    mid_point = (nodes[ltgi(element,0)]+nodes[ltgi(element,2)])/2
    jac=jacobian_global_to_local_map(nodes[ltgi(element,0)],nodes[ltgi(element,2)])
    A[ltgi(element,0)][ltgi(element,0)] += sym.integrate(shape_derivative_0*shape_derivative_0,(x,0,1))/jac
    A[ltgi(element,1)][ltgi(element,0)] += sym.integrate(shape_derivative_1*shape_derivative_0,(x,0,1))/jac
    A[ltgi(element,2)][ltgi(element,0)] += sym.integrate(shape_derivative_2*shape_derivative_0,(x,0,1))/jac
    A[ltgi(element,0)][ltgi(element,1)] += sym.integrate(shape_derivative_0*shape_derivative_1,(x,0,1))/jac
    A[ltgi(element,1)][ltgi(element,1)] += sym.integrate(shape_derivative_1*shape_derivative_1,(x,0,1))/jac
    A[ltgi(element,2)][ltgi(element,1)] += sym.integrate(shape_derivative_2*shape_derivative_1,(x,0,1))/jac
    A[ltgi(element,0)][ltgi(element,2)] += sym.integrate(shape_derivative_0*shape_derivative_2,(x,0,1))/jac
    A[ltgi(element,1)][ltgi(element,2)] += sym.integrate(shape_derivative_1*shape_derivative_2,(x,0,1))/jac
    A[ltgi(element,2)][ltgi(element,2)] += sym.integrate(shape_derivative_2*shape_derivative_2,(x,0,1))/jac
    #integrate with simpson's method.
    M = f.subs(x,mid_point)*shape_function_0.subs(x,0.5)*jac
    T = (f.subs(x,nodes[ltgi(element,0)])*shape_function_0.subs(x,0.0)+
    f.subs(x,nodes[ltgi(element,2)])*shape_function_0.subs(x,1.0))*jac*0.5
    f_d[ltgi(element,0)] = f_d[ltgi(element,0)] + (2*M+T)/3
    M = f.subs(x,mid_point)*shape_function_1.subs(x,0.5)*jac
    T = (f.subs(x,nodes[ltgi(element,0)])*shape_function_1.subs(x,0.0)+
    f.subs(x,nodes[ltgi(element,2)])*shape_function_1.subs(x,1.0))*jac*0.5
    f_d[ltgi(element,1)] = f_d[ltgi(element,1)] + (2*M+T)/3
    M = f.subs(x,mid_point)*shape_function_2.subs(x,0.5)*jac
    T = (f.subs(x,nodes[ltgi(element,0)])*shape_function_2.subs(x,0.0)+
    f.subs(x,nodes[ltgi(element,2)])*shape_function_2.subs(x,1.0))*jac*0.5
    f_d[ltgi(element,2)] = f_d[ltgi(element,2)] + (2*M+T)/3

# Go through boundary
# Impose Dirichlet condition
A[0][0] = 1
A[0][1:num_nodes-1] = 0
f_d[0] = u_0
# Impose Neumann condition
f_d[num_nodes-1] = f_d[num_nodes-1]+u_prime_n
#Solve linear system
u_h = np.linalg.solve(A,f_d)
#Plot the solution using matplotlib
plot_fe = plt.plot(nodes,u_h)
#We can also calculate the error at the nodes (which should only be accumulated in the quadrature
error_fe = np.amax(np.absolute(u_h-u_fabric_nodes))
print('Max error at nodes: ',error_fe)
#Calculate L2 norm with midpoint rule on the interpolated solution using lagrange polynomials.
error_l2 = 0
for i in range(num_elements):
    x_vec = np.array([nodes[ltgi(i,0)],nodes[ltgi(i,1)],nodes[ltgi(i,2)]])
    y_vec = np.array([u_h[ltgi(i,0)],u_h[ltgi(i,1)],u_h[ltgi(i,2)]])
    poly = lagrange(x_vec,y_vec)
    mid_point = (nodes[ltgi(i,0)]+nodes[ltgi(i,1)])/2
    mid_u1 = np.polyval(poly,mid_point)
    area = nodes[ltgi(i,1)]-nodes[ltgi(i,0)]
    error_l2 += (u_fabric.subs(x,mid_point)-mid_u1)**2*area
    mid_point = (nodes[ltgi(i,1)]+nodes[ltgi(i,2)])/2
    mid_u2 = np.polyval(poly,mid_point)
    area = nodes[ltgi(i,2)]-nodes[ltgi(i,1)]
    error_l2 += (u_fabric.subs(x,mid_point)-mid_u2)**2*area
error_l2 = math.sqrt(error_l2)
print("L2 error:", error_l2)
plt.plot(nodes,u_h)
plt.plot(nodes,u_fabric_nodes)
plt.show()
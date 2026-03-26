#Param study 001

import json
import numpy as np
import matplotlib.pyplot as plt

from Frame_2D import Model

model = Model.from_json("param_01.json")
print(f"problem_type: {model.problem_type}")

#shift of node 3 and 4
shift = np.arange(0, 18)

#Results
A=np.zeros(len(shift))
B=np.zeros(len(shift))
x3=np.zeros(len(shift))
x4=np.zeros(len(shift))
M3=np.zeros(len(shift))
M4=np.zeros(len(shift))
V31=np.zeros(len(shift))
V34=np.zeros(len(shift))
V43=np.zeros(len(shift))
V42=np.zeros(len(shift))
x3_0=model.nodes[3].x
x4_0=model.nodes[4].x

for i, s in enumerate(shift):
    #modification
    x3[i]=x3_0 + s
    x4[i]=x4_0 + s
    model.nodes[3].x = x3[i]
    model.nodes[4].x = x4[i]

    #solve
    model.solve()

    #results
    A[i] = model.reaction_in_dof(2)
    B[i] = model.reaction_in_dof(5)
    M3[i] = model.element_end_forces(model.elements[1], displacement_vector=None)[5]
    M4[i] = model.element_end_forces(model.elements[2], displacement_vector=None)[5]
    V31[i] = - model.element_end_forces(model.elements[1], displacement_vector=None)[4]
    V34[i] = + model.element_end_forces(model.elements[2], displacement_vector=None)[1]
    V43[i] = - model.element_end_forces(model.elements[2], displacement_vector=None)[4]
    V42[i] = + model.element_end_forces(model.elements[3], displacement_vector=None)[1]

print("shift:" ,shift)
print("A:" ,A)
print("B:" ,B)
print("M3:" ,M3)
print("M4:" ,M4)
    
#Graph
plt.figure()
for i in range(len(shift)):
    X = np.array([0, x3[i], x4[i], model.nodes[2].x])
    Y = np.array([0, M3[i], M4[i], 0])   
    plt.plot(X, Y)
plt.xlabel("x [m]")
plt.ylabel("M [kNm]")
# invert Y 
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

plt.figure()
for i in range(len(shift)):
    X = np.array([0, x3[i], x3[i], x4[i], x4[i], model.nodes[2].x])
    Y = np.array([A[i], V31[i], V34[i],  V43[i],  V42[i],  -B[i]])
    plt.plot(X, Y)
plt.xlabel("x [m]")
plt.ylabel("V [kN]")
plt.grid(True)
plt.show() 










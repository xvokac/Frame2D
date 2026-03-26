#Param study 001

import json
import numpy as np
import matplotlib.pyplot as plt

from Frame_2D import Model

model = Model.from_json("param_02.json")
print(f"problem_type: {model.problem_type}")

#nodal load factor
alpha_F = np.arange(0, 15)
alpha_F = alpha_F * 2

#Results
x=np.zeros(len(alpha_F))
M=np.zeros(len(alpha_F))
F=model.nodal_loads[0].Fy

for i, a in enumerate(alpha_F):
    #modification
    model.nodal_loads[0].Fy = F * alpha_F [i] 

    #solve
    model.solve()

    #results - extreme of M
    x[i], M[i] = model.element_internal_force_extreme(model.elements[1], "M", "max")
    
print("=== M(x) extremes ===")
print("alpha_F:" ,alpha_F)
print("x [m]:" ,x)
print("M [kNm]:" ,M)






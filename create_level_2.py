# %%
import numpy as np

def create_level(level_nbr): 
    
    A = np.ones((7,12))
    for i in range(0,7):
        A[i,0] = 2
        A[i,11] = 2
    for j in range(0,12):
        A[0,j] = 3
    
    filename = 'Levels/level'+str(level_nbr)+'.csv'   
    with open(filename, "w") as f:
        np.savetxt(f,A)


create_level(666)
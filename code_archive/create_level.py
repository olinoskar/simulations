import numpy as np

def create_level(level_nbr): 
    
    A = np.random.rand(7,12) + 0.2
    A = np.floor(A)
    
    filename = 'Levels/level'+str(level_nbr)+'.csv'   
    with open(filename, "w") as f:
        np.savetxt(f,A)


#create_level(5)

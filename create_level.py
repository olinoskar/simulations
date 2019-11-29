import numpy as np

def create_level(const): 
    
    A = np.random.rand(7,12) + const + 0.2
    A = np.floor(A)
    
    filename = 'Levels/level.csv'   
    with open(filename, "w") as f:
        np.savetxt(f,A)
        
    #filename = 'ArchivedLevels/level.csv'        
    #with open(filename, "w") as f:
    #np.savetxt(f,A)
        
#%% Read and print
#data = np.genfromtxt(filename)
#print(data)


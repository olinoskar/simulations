#%% One way of playing the game

from Network import Network
from dxball import play_game
network = Network([77,3,3])

#network.load(path='results/test_run2')
    
score, frames_run, fitness = play_game(network,use_network=0,
                                       course_nbr=666,display_game=1,fps=50,
                                       max_nbr_frames=20000,score_exponent=0.35, 
                                       frame_exponent=0.35)
#%% One way of playing the game


from dxball import play_game
network = []
score, frames_run, fitness = play_game(network)

#%% Another way of playing the game
score, frames_run, fitness = play_game(network,use_network=1,
                                       course_nbr=666,display_game=1,fps=50,
                                       max_nbr_frames=1000,score_exponent=0.35, 
                                       frame_exponent=0.35) 
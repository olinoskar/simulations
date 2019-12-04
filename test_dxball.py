# %% Run the game and return score
from dxball import play_game

network = []
max_playtime = 10  # seconds
display_game = 1
fps = 50
max_nbr_frames = 1000  # some arbitrary number
course_nbr = 666

score, frames_run = play_game(network,course_nbr,max_playtime,display_game,fps,max_nbr_frames)
frames_str = "{:.2f}".format(frames_run)
print("Score = " + str(score) + " after " + frames_str + " frames played.") 


# %% Run the game and return score
from dxball import play_game

network = []
course = []
max_playtime = 10  # seconds
display_game = 1
fps = 100

score, playtime = play_game(network,course,max_playtime,display_game,fps)
time_str = "{:.2f}".format(playtime)
print("Score = " + str(score) + " after " + time_str + " seconds played.") 


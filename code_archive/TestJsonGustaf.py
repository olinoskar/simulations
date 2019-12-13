#%%
import json
import pprint

with open('params.json', 'r') as f:
    text = f.read()
    
params_list = json.loads(text)
params_list[0]
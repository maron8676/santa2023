import pickle
import tqdm

with open('cube-222c.pkl', mode='rb') as f:
    cube222 = pickle.load(f)

key_list = list(cube222.keys())
reformat = {}
for key in tqdm.tqdm(key_list):
    reformat[';'.join(list(key))] = cube222[key]

with open('cube-222c.pkl', mode='wb') as f:
    pickle.dump(reformat, f)

import pickle
import tqdm

with open('cube-222b.pkl', mode='rb') as f:
    cube222 = pickle.load(f)

key_list = list(cube222.keys())
reformat = {}
alpha = {v: k for k, v in enumerate('ABCDEFGHIJKLMNOPQRSTUVWX')}
for key in tqdm.tqdm(key_list):
    key2 = list(key)
    key2 = list(map(lambda x: 'N' + str(alpha[x]), key2))
    reformat[';'.join(list(key2))] = cube222[key]

with open('cube-222b.pkl', mode='wb') as f:
    pickle.dump(reformat, f)

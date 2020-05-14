
# Given a random obj id, I need to find a scene with said object.
# Therefore I need a dictionary that, given an obj id, returns a scene id.
# I couldn't find any list of which scenes contains which objects, so I'll have 
# to scrape the gt.yml files.
import os
import ruamel.yaml as yaml
import numpy as np
import pickle 


tlesspath = '/home/grans/Documents/t-less_v2/test_primesense/'
scenes = list(range(1, 21))

scene2objlist = {scene: None for scene in list(range(1, 21))}
obj2scenelist = {obj: [] for obj in list(range(1, 31))}

for scene in scenes:
    gtfile = os.path.join(tlesspath, '{:02d}', 'gt.yml').format(scene)
    with open(gtfile, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        obj_ids = np.unique([d['obj_id'] for d in gts[0]])
        [obj2scenelist[obj_id].append(scene) for obj_id in obj_ids]
        scene2objlist[scene] = obj_ids
        print(str(scene) + ": " + str(obj_ids))


print("hold")
pickle.dump(scene2objlist, open('scene2objlist.pkl', 'wb'))
pickle.dump(obj2scenelist, open('obj2scenelist.pkl', 'wb'))


import numpy as np
from glob import glob


np_list = glob('/media/datadrive/IITB_corridor/frame_mask/*.npy')
np_list.sort()
print(np_list)

gt = []
for npy in np_list:
    gt.append(np.load(npy))

print(gt)

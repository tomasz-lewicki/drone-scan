import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

np.random.seed(42) # reproducibility

def ignite_center(state):
    center_x = int(state.shape[0]/2)
    center_y = int(state.shape[1]/2)
    state[center_x-3:center_x+3,center_y-3:center_y+3] = 1
    

def gkern(l=5, sig=1.):
    """    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


CELLS = 1000
TREE_DENSITY = 0.55
BURN_RATE = 5 # 0-255
N_STEPS = 1000
IGNITION_PROB = 0.25


n_trees = int(CELLS**2 * TREE_DENSITY) # fill 55% with trees. With 3x3 kernel the fire doesn't spread at 50% rate

trees = np.random.randint(0,CELLS,(n_trees,2)) # choose 50000 tree locations (assign fuel to half of cells)
fuel = np.zeros((CELLS,CELLS), dtype=np.uint8) # could be 1000x1000m
fuel[trees[:,0], trees[:,1]] = np.random.randint(0,255,n_trees)

# state = np.array(np.zeros_like(fuel), dtype=np.bool)
# ignite_center(state)

kernel3by3 = np.array([[1,1,1],[1,0,1],[1,1,1]]) # 3x3
kernel5by5 = 15 *gkern()
kernel5by5[2,2] = 0
kernel5by5 = kernel3by3

state = np.zeros_like(fuel)
ignite_center(state)
states_history = np.zeros((N_STEPS,*state.shape))
ignitions_history = np.zeros((N_STEPS,*state.shape))
for i in range(N_STEPS):

    states_history[i] = state
    
    # calculate new ignitons
    count_neighbors_on_fire = scipy.signal.convolve2d(state, kernel5by5, 'same')
    ignitions = (count_neighbors_on_fire * np.random.random(state.shape) > IGNITION_PROB) * fuel
    state += ignitions
    
    ignitions_history[i] = ignitions

    #update fuel status
    on_fire_mask = state > 0
    burned_out_mask = fuel < BURN_RATE
    fuel[on_fire_mask] -= BURN_RATE 
    fuel[burned_out_mask] = 0
    state[burned_out_mask] = 0

import skimage.io
import os
import time

name =  f'cells={str(CELLS)}steps={str(N_STEPS)}ignition_prob={str(IGNITION_PROB)}tree_density={str(TREE_DENSITY)}'
os.mkdir(name)

for i in range(N_STEPS):
    padded_n = '{0:04d}'.format(i)
    skimage.io.imsave(f'{name}/frame{padded_n}.png',states_history[i])


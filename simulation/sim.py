import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import skimage.io
import os
import cv2

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

CELLS = 10000
TREE_DENSITY = 0.52512
BURN_RATE = 2 # 0-255
N_STEPS = 10
IGNITION_PROB = 0.20
N_EPOCHS = 10
COLOR = False

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

states_history = np.array(np.zeros((N_STEPS,*state.shape)), dtype=np.uint8)
ignitions_history = np.zeros((N_STEPS,*state.shape), dtype=np.uint8)
fuel_history = np.zeros((N_STEPS,*state.shape), dtype=np.uint8)

name =  f'cells={str(CELLS)}steps={str(N_EPOCHS*N_STEPS)}ignition_prob={str(IGNITION_PROB)}tree_density={str(TREE_DENSITY)}burn_rate={str(BURN_RATE)}'
os.mkdir(name)

for e in range(N_EPOCHS):
    print(f'epoch:{e}')
    # run simulation
    for i in range(N_STEPS):


        # calculate new ignitons
        count_neighbors_on_fire = scipy.signal.convolve2d(state, kernel5by5, 'same')
        ignitions = (count_neighbors_on_fire * np.random.random(state.shape) > IGNITION_PROB) * fuel
        state += ignitions
        

        #update fuel status
        on_fire_mask = state > 0
        burned_out_mask = fuel < BURN_RATE
        fuel[on_fire_mask] -= BURN_RATE 
        fuel[burned_out_mask] = 0
        state[burned_out_mask] = 0

        # update histories
        ignitions_history[i] = ignitions
        states_history[i] = state
        fuel_history[i] = fuel

    # save images
    for i in range(N_STEPS):
        padded_n = '{0:05d}'.format(e*N_STEPS+i)

        red = cv2.resize(states_history[i], (2000,2000))
        if COLOR:
            green = cv2.resize(fuel_history[i], (2000,2000))
            blue = np.zeros_like(green)
            im = np.stack([red, green, blue], axis=-1)
        else:
            im = red

        # skimage.io.imsave(f'{name}/frame{padded_n}.png', im)

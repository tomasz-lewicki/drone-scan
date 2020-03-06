#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

from threading import Thread
from simulation import run, make_fuel_map, ignite_center, state_to_rgb
from geometry import Space

# to convert the output png files:
# ffmpeg -i out%03d.png -c:v libx264 -crf 0 -preset veryslow -c:a libmp3lame -b:a 320k output.mp4

OUT_DIR_NAME = 'drones_positions_plt'
N_DRONES = 10
N_CELLS = 100
VELOCITY = 5

# sin_cos_heading = np.hstack([np.sin(drones_heading), np.cos(drones_heading)])
# np.sum(np.power(sin_cos_heading,2), axis=1) # sanity check, should be all ones

if __name__ == '__main__':

    ######################################################################
    # Create Environment
    fuel = make_fuel_map(N_CELLS, tree_density=0.55)
    wildfire_state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(wildfire_state)

    # define space (just a convinience wrapper for coordinate transforms)
    # sp = Space(origin=(37.331553, -121.882767)) # south-west corner of campus
    sp = Space(origin=(37.335404, -121.883400)) # fountain

    ######################################################################
    # Simulation
    sim_args = {
        'state_array': wildfire_state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 100,
        'ignition_prob': 0.35,
        'n_epochs': 10,
        'save_dir': None
        # 'save_dir' = f"sim_output_cells={str(wildfire_state.shape)}_steps={str(sim_args['n_epochs']*sim_args['n_steps'])}_ignition_prob={str(sim_args['ignition_prob'])}_burn_rate={str(sim_args['burn_rate'])}_time={str(datetime.datetime.now())}"
    }

    # we will output images here
    if sim_args['save_dir'] is not None and (not os.path.isdir(sim_args['save_dir'])):
        os.mkdir(sim_args['save_dir'])

    # run simulation in a separate thread
    sim_thr = Thread(target=run, kwargs=sim_args)
    sim_thr.start()

    ######################################################################
    # Drones !

    global_belief = np.zeros_like(wildfire_state)

    # init location matrix
    drones_xy = np.random.random((N_DRONES,2)) * N_CELLS

    # generate heading for all drones
    drones_heading = np.random.random((N_DRONES,1)) * 360

    # init velocity matrix
    drones_velocity_xy = VELOCITY * np.hstack([np.sin(drones_heading), np.cos(drones_heading)])

    if not os.path.isdir(OUT_DIR_NAME):
        os.mkdir(OUT_DIR_NAME)

    plt.figure()
    ax = plt.axes()



    for step in range(1000):
        drones_xy += drones_velocity_xy
        
        for i, ((x,y),(vx,vy)) in enumerate(zip(drones_xy, drones_velocity_xy)):
            if x<0 or x>N_CELLS:
                drones_velocity_xy[i,0] = -vx
            if y<0 or y>N_CELLS:
                drones_velocity_xy[i,1] = -vy
        

        for (x,y) in drones_xy:
            x,y = int(x), int(y)
            x = max(0, min(x,N_CELLS-1))
            y = max(0, min(y,N_CELLS-1))

            global_belief[x-5:x+5][y-5:y+5] = wildfire_state[x-5:x+5][y-5:y+5]

        # plot stuff
        if step % 1 == 0:

            plt.clf()
            plt.xlim([-N_CELLS/5, 1.2*N_CELLS])
            plt.ylim([-N_CELLS/5, 1.2*N_CELLS])
            #im = state_to_rgb(wildfire_state, fuel)
            # plt.imshow(im)
            plt.imshow(global_belief)
            plt.scatter(drones_xy[:,0], drones_xy[:,1], color='white')
            #ax = plt.axes()
            
            # plot vectors
            for (x,y),(vx,vy) in zip(drones_xy, drones_velocity_xy):
                ax.arrow(x, y, 3*vx, 3*vy, head_width=0.05, head_length=1, fc='k', ec='k')
            
            plt.show(block=False)
            plt.pause(.000001)
            plt.savefig(f'{OUT_DIR_NAME}/out{step:03d}.png', dpi=300)

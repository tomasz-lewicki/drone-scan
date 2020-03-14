#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from threading import Thread
from simulation import run, make_fuel_map, ignite_center, state_to_rgb

# to convert the output png files:
# ffmpeg -i out%03d.png -c:v libx264 -crf 0 -preset veryslow -c:a libmp3lame -b:a 320k output.mp4

OUT_DIR_NAME =  f"sim_output_time={str(datetime.datetime.now())}"

if not os.path.isdir(OUT_DIR_NAME):
    os.mkdir(OUT_DIR_NAME)

N_DRONES = 100
N_CELLS = 300
VELOCITY = 50
FOV = 15 # field of view for individual drones

GUI_MARGIN = 0.1

# sin_cos_heading = np.hstack([np.sin(drones_heading), np.cos(drones_heading)])
# np.sum(np.power(sin_cos_heading,2), axis=1) # sanity check, should be all ones
np.random.seed(420)

if __name__ == '__main__':

    ######################################################################
    # Create Environment
    fuel = make_fuel_map(N_CELLS, tree_density=0.6) #tree_density 0.45-0.55 is an interesting case
    wildfire_state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(wildfire_state)

    ######################################################################
    # Simulation
    sim_args = {
        'state_array': wildfire_state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 100,
        'ignition_prob': 0.35,
        'n_epochs': 10,
        'save_dir': None,
        'loop_min_dur': 0.1, #throttle down to max 10FPS
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
    visited = np.zeros_like(wildfire_state)

    # init location matrix
    drones_xy = np.random.random((N_DRONES,2)) * N_CELLS

    # generate heading for all drones
    drones_heading = np.random.random((N_DRONES,1)) * 360

    # init velocity matrix
    drones_velocity_xy = VELOCITY * np.hstack([np.sin(drones_heading), np.cos(drones_heading)])

    if not os.path.isdir(OUT_DIR_NAME):
        os.mkdir(OUT_DIR_NAME)

    fig, axs = plt.subplots(ncols=2, figsize=(15,15))
    
    gui_iter = 0
    while(sim_thr.is_alive()):
        gui_iter += 1
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

            global_belief[y-FOV:y+FOV,x-FOV:x+FOV] = wildfire_state[y-FOV:y+FOV,x-FOV:x+FOV]
            visited[y-FOV:y+FOV,x-FOV:x+FOV] = 255 

        # plot stuff
        if gui_iter % 1 == 0:
            
            axs[1].set(
                xlim=(-N_CELLS*GUI_MARGIN, (1+GUI_MARGIN)*N_CELLS),
                ylim=(-N_CELLS*GUI_MARGIN, (1+GUI_MARGIN)*N_CELLS))

            # Ground truth axis
            im = state_to_rgb(wildfire_state, fuel)
            axs[1].imshow(im)

            # Belief axis
            axs[0].clear()
            im = state_to_rgb(global_belief, np.zeros_like(global_belief))
            
            im = np.stack(3*[global_belief], axis=-1)
            axs[0].imshow(im)
            axs[0].scatter(drones_xy[:,0], drones_xy[:,1], color='yellow', s=len(drones_xy[:,0])*[5])
            axs[0].set(
                xlim=(-N_CELLS*GUI_MARGIN, (1+GUI_MARGIN)*N_CELLS),
                ylim=(-N_CELLS*GUI_MARGIN, (1+GUI_MARGIN)*N_CELLS))
            
            # plot velocity vectors
            # for (x,y),(vx,vy) in zip(drones_xy, drones_velocity_xy):
            #     axs[0].arrow(x, y, 3*vx, 3*vy, head_width=0.05, head_length=.5, fc='y', ec='y')
            
            plt.show(block=False)
            plt.pause(.000001)

            # We can safe the output (optional)
            plt.savefig(f'{OUT_DIR_NAME}/out{gui_iter:03d}.png', dpi=300) 
 
    sim_thr.join()
    plt.close('all')

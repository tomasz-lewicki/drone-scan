from multiprocessing import Process
from threading import Thread

import numpy as np
import datetime, time
import os

from simulation.sim import run, make_fuel_map, ignite_center
if __name__ == '__main__':

    N_CELLS = 100
    N_DRONES = 10
    
    #create fuel map
    fuel = make_fuel_map(N_CELLS, tree_density=0.55)

    # create array holding the states of the simulation
    state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(state)

    sim_args = {
        'state_array': state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 10,
        'ignition_prob': 0.2,
        'n_epochs': 10
    }

    # we will output images here
    dir_name =  f"sim_output_cells={str(state.shape)}_steps={str(sim_args['n_epochs']*sim_args['n_steps'])}_ignition_prob={str(sim_args['ignition_prob'])}_burn_rate={str(sim_args['burn_rate'])}_time={str(datetime.datetime.now())}"
    
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    sim_args['save_dir'] = dir_name

    p = Thread(target=run, kwargs=sim_args)
    p.start()

    for i in range(100):
        time.sleep(0.5)
        print(state[45:55,45:55])

    p.join()
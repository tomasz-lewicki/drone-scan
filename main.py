from multiprocessing import Process
from threading import Thread
import asyncio

import numpy as np
import datetime, time
import os
from simulation.sim import run, make_fuel_map, ignite_center
from drone_routines import Drone, Space

from mavsdk import start_mavlink
from mavsdk import connect as mavsdk_connect
from mavsdk import (MissionItem)

N_CELLS = 100

if __name__ == '__main__':

    # Drone
    drone = Drone()

    # Create Environment
    # fuel map
    fuel = make_fuel_map(N_CELLS, tree_density=0.55)

    # create array holding the states of the simulation
    wildfire_state = np.zeros_like(fuel) # should the type be dtype=np.bool?
    ignite_center(wildfire_state)

    belief = []

    # define space
    # sp = Space(origin=(37.331553, -121.882767)) # south-west corner of campus
    sp = Space(origin=(37.335404, -121.883400)) # fountain
    
    # generate trajectory
    traj = []
    step = 50 # [grid step of 20m]
    for x in range(0,N_CELLS,step):
        traj.append((x,0))
        traj.append((x,N_CELLS)) # that's assuming N_CELLS is equal to search space size
        traj.append((x+step/2,N_CELLS))
        traj.append((x+step/2,0))

    # get trajectory in global frame
    traj_global = list(map(lambda p: sp.xy2latlon(*p), traj))

    # Simulation
    sim_args = {
        'state_array': wildfire_state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 10,
        'ignition_prob': 0.2,
        'n_epochs': 10
    }

    # we will output images here
    sim_args['save_dir'] =  f"sim_output_cells={str(wildfire_state.shape)}_steps={str(sim_args['n_epochs']*sim_args['n_steps'])}_ignition_prob={str(sim_args['ignition_prob'])}_burn_rate={str(sim_args['burn_rate'])}_time={str(datetime.datetime.now())}"
    
    if not os.path.isdir(sim_args['save_dir']):
        os.mkdir(sim_args['save_dir'])

    # run simulation

    sim_thr = Thread(target=run, kwargs=sim_args)
    sim_thr.start()

    # do the drone stuff
    asyncio.ensure_future(drone.run_scan(traj_global))
    asyncio.ensure_future(drone.print_mission_progress())
    asyncio.ensure_future(drone.print_battery())
    asyncio.ensure_future(drone.measure(belief, wildfire_state))
    asyncio.get_event_loop().run_until_complete(drone.observe_is_in_air())

    sim_thr.join()
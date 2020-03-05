from multiprocessing import Process
from threading import Thread
import asyncio

import numpy as np
import datetime, time
import os
import matplotlib.pyplot as plt
from simulation.sim import run, make_fuel_map, ignite_center
from drone_routines import Drone, Space

from mavsdk import start_mavlink
from mavsdk import connect as mavsdk_connect
from mavsdk import (MissionItem)

N_CELLS = 100

async def print_gui(wildfire_array):
    while(True):
        plt.plot(np.random.rand(2,10))
        plt.show(block=False)
        plt.pause(.00001)
        await asyncio.sleep(1)

# Before starting:
# docker run --rm -it --env PX4_HOME_LAT=37.335404 --env PX4_HOME_LON=-121.883400 --env PX4_HOME_ALT=488.0 jonasvautherin/px4-gazebo-headless:v1.9.2
# mavproxy.py --master=udp:0.0.0.0:14550 --out=udp:localhost:14551 --out=udp:localhost:14552


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

    # define space (convinience wrapper for coordinate transforms)
    # sp = Space(origin=(37.331553, -121.882767)) # south-west corner of campus
    sp = Space(origin=(37.335404, -121.883400)) # fountain
    
    # generate trajectory
    traj = []
    grid_res_x = 10
    grid_len_x = 30 # [grid step of 20m]
    grid_len_y = 20

    for x in range(0, grid_len_x, grid_res_x):
        traj.append((x,0))
        traj.append((x,grid_len_y)) # that's assuming N_CELLS is equal to search space size
        traj.append((x+grid_res_x/2,grid_len_y))
        traj.append((x+grid_res_x/2,0))

    # get trajectory in global frame
    traj_global = list(map(lambda p: sp.xy2latlon(*p), traj))

    # Simulation
    sim_args = {
        'state_array': wildfire_state,
        'fuel_array': fuel,
        'burn_rate': 3,
        'n_steps': 100,
        'ignition_prob': 0.3,
        'n_epochs': 10
    }

    # we will output images here
    sim_args['save_dir'] =  f"sim_output_cells={str(wildfire_state.shape)}_steps={str(sim_args['n_epochs']*sim_args['n_steps'])}_ignition_prob={str(sim_args['ignition_prob'])}_burn_rate={str(sim_args['burn_rate'])}_time={str(datetime.datetime.now())}"
    
    if not os.path.isdir(sim_args['save_dir']):
        os.mkdir(sim_args['save_dir'])

    # run simulation in a separate thread
    sim_thr = Thread(target=run, kwargs=sim_args)
    sim_thr.start()

    # do the drone stuff
    # will it work x30 drones??

    # asyncio.ensure_future(drone.run_scan(traj_global)) 
    # asyncio.ensure_future(drone.print_mission_progress())
    asyncio.ensure_future(drone.print_pos(belief))
    
    asyncio.ensure_future(print_gui(wildfire_state))

    #asyncio.ensure_future(drone.measure(belief=belief, space_transform=sp, ground_truth=wildfire_state))
    asyncio.get_event_loop().run_until_complete(drone.observe_is_in_air())

    # wait for the sim thread to finish
    
    # plot trajectory
    traj_x = list(map(lambda b: b[0][1], belief))
    traj_y = list(map(lambda b: b[0][1], belief))
    fires = list(map(lambda b: b[1], belief))

    plt.scatter(traj_x, traj_y, s=0.1)
    # plt.show()

    sim_thr.join()
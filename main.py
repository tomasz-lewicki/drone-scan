import asyncio
import numpy as np
import datetime, time
import os
import matplotlib.pyplot as plt

from threading import Thread
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

def create_grid_trajectory(x_len=30, y_len=20, x_res=10):
    # generate trajectory

    traj = []
    for x in range(0, x_len, x_res):
        traj.append((x,0))
        traj.append((x,y_len))
        traj.append((x+x_res/2,y_len))
        traj.append((x+x_res/2,0))

    # get trajectory in global frame
    return list(map(lambda p: sp.xy2latlon(*p), traj))

# Before starting:
# docker run --rm -it --env PX4_HOME_LAT=37.335404 --env PX4_HOME_LON=-121.883400 --env PX4_HOME_ALT=488.0 jonasvautherin/px4-gazebo-headless:v1.9.2
# mavproxy.py --master=udp:0.0.0.0:14550 --out=udp:localhost:14551 --out=udp:localhost:14552

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
        'ignition_prob': 0.3,
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
    # Drone
    # TODO: will this work for 100 drones?

    belief = []
    traj_global = create_grid_trajectory(x_len=50, y_len=10, x_res=10)
    
    drone = Drone()
    # asyncio.ensure_future(drone.run_scan(traj_global)) 
    # asyncio.ensure_future(drone.print_mission_progress())
    asyncio.ensure_future(drone.print_pos(belief))
    # asyncio.ensure_future(drone.measure(belief=belief, space_transform=sp, ground_truth=wildfire_state))
    asyncio.get_event_loop().run_until_complete(drone.observe_is_in_air())

    ######################################################################
    # GUI
    asyncio.ensure_future(print_gui(wildfire_state))
    
    # plot trajectory
    # traj_x = list(map(lambda b: b[0][1], belief))
    # traj_y = list(map(lambda b: b[0][1], belief))
    # fires = list(map(lambda b: b[1], belief))
    # plt.scatter(traj_x, traj_y, s=0.1)
    # plt.show()

    sim_thr.join()
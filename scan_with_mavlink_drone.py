import asyncio
import numpy as np
import datetime, time
import os
import matplotlib.pyplot as plt
import cv2
import copy

from threading import Thread
from simulation import run, make_fuel_map, ignite_center, state_to_rgb
from drone_routines import Drone
from geometry import Space


SPACE_SIZE_X = 50 # space size in meters
SPACE_SIZE_Y = 50
N_CELLS = 100

async def print_gui(fire_arr, space, fuel_arr, belief):
    # could be instead iterating over an async belief generator
    while(True):
        # measurements = np.array([((*b[0]), b[1]) for b in belief])
        if len(belief) > 0:
            (x,y) = belief[-1][0]
            (lat, lon) = space.xy2latlon(x,y)

            cell_size_x = SPACE_SIZE_X/N_CELLS
            cell_size_y = SPACE_SIZE_Y/N_CELLS
            cell_x, cell_y = x/cell_size_x, y/cell_size_y
            img = state_to_rgb(fire_arr, fuel_arr)

            plt.imshow(img)
            
            # plt.imshow(fire_arr)
            plt.scatter(cell_x, cell_y, c='white')
            plt.show(block=False)
            plt.pause(.00001)

        await asyncio.sleep(.1)

def create_grid_trajectory(x_len=30, y_len=30, x_res=10):
    # generate grid trajectory as a list of tuples in local frame(x,y)

    traj = []
    for x in range(0, x_len, x_res):
        traj.append((x,0))
        traj.append((x,y_len))
        traj.append((x+x_res/2,y_len))
        traj.append((x+x_res/2,0))

    return traj

async def plot_drone_position(drone):
    async for pos in drone.telemetry.position():

        #measurements = np.array([((*b[0]), b[1]) for b in belief])
        #plt.scatter(measurements[:,0], measurements[:,1], c=measurements[:,2])
        plt.scatter(pos[0],pos[1])
        plt.show(block=False)
        plt.pause(.00001)
        await asyncio.sleep(1)

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
        'ignition_prob': 0.25,
        'n_epochs': 10,
        'save_dir': None,
        'loop_min_dur': 1 # throttle to 1FPS
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
    
    loop = asyncio.get_event_loop()
    drone = Drone()

    loop.run_until_complete(drone._drone.connect(system_address=drone._url))

    belief = []

    traj_local = create_grid_trajectory(x_len=SPACE_SIZE_X, y_len=SPACE_SIZE_Y, x_res=10)
    traj_global = list(map(lambda p: sp.xy2latlon(*p), traj_local))


    asyncio.ensure_future(drone.run_scan(traj_global)) 
    asyncio.ensure_future(drone.print_mission_progress())
    asyncio.ensure_future(drone.print_pos(belief))
    asyncio.ensure_future(print_gui(wildfire_state, sp, fuel, belief))
    asyncio.ensure_future(drone.measure(belief=belief, space_transform=sp, ground_truth=wildfire_state))

    loop.run_forever()

    ######################################################################
    # GUI
    
    # plot trajectory
    # traj_x = list(map(lambda b: b[0][1], belief))
    # traj_y = list(map(lambda b: b[0][1], belief))
    # fires = list(map(lambda b: b[1], belief))
    # plt.scatter(traj_x, traj_y, s=0.1)
    # plt.show()

    sim_thr.join()
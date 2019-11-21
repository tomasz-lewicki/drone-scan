import asyncio

from mavsdk import start_mavlink
from mavsdk import connect as mavsdk_connect
from mavsdk import (MissionItem)

import numpy as np
from geopy import Point

import geopy, geopy.distance
import math

import math
from math import cos

import matplotlib.pyplot as plt

class Space:
    
    def __init__(self, origin):
        self.lat0 = origin[0]
        self.lon0 = origin[1]
        
        lat0rad = math.radians(self.lat0)
        self.mdeglon = (111415.13 * cos(lat0rad)
                        - 94.55 * cos(3.0*lat0rad)
                        - 0.12 * cos(5.0*lat0rad) )
        
        lat0rad = math.radians(self.lat0)
        self.mdeglat = (111132.09 - 566.05 * cos(2.0*lat0rad)
                        + 1.20 * cos(4.0*lat0rad)
                        - 0.002 * cos(6.0*lat0rad))
        
    # method taken from: https://www.researchgate.net/publication/224239584_Rectilinear_coordinate_frames_for_Deep_sea_navigation
    def latlon2xy(self, lat, lon):
        x = (lon-self.lon0) * self.mdeglon
        y = (lat-self.lat0) * self.mdeglat
        return (x, y)
    
    def xy2latlon(self, x, y):
        lon = x/self.mdeglon + self.lon0
        lat = y/self.mdeglat + self.lat0
        return (lat, lon)
    
async def run(traj):

    mission_items = []

    for p in traj:
        mission_items.append(MissionItem(p[0],
                                        p[1],
                                        25,
                                        10,
                                        True,
                                        float('nan'),
                                        float('nan'),
                                        MissionItem.CameraAction.NONE,
                                        float('nan'),
                                        float('nan')))
                                     

    await drone.mission.set_return_to_launch_after_mission(True)

    print("-- Uploading mission")
    await drone.mission.upload_mission(mission_items)

    print("-- Arming")
    await drone.action.arm()

    print("-- Starting mission")
    await drone.mission.start_mission()


async def print_mission_progress():
    async for mission_progress in drone.mission.mission_progress():
        print(f"Mission progress: {mission_progress.current_item_index}/{mission_progress.mission_count}")


async def observe_is_in_air():
    """ Monitors whether the drone is flying or not and
    returns after landing """

    was_in_air = False

    async for is_in_air in drone.telemetry.in_air():
        if is_in_air:
            was_in_air = is_in_air

        if was_in_air and not is_in_air:
            await asyncio.get_event_loop().shutdown_asyncgens()
            return

async def print_battery(drone):
    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent}")

async def measure(drone):
    async for position in drone.telemetry.position():
        global belief, truth
        # sample the truth in current position & append to belief
        coords = sp.latlon2xy(position.latitude_deg, position.longitude_deg)
        belief.append((coords, truth[int(coords[0]), int(coords[1])]))

async def plot_belief(drone):
    async for pos in drone.telemetry.position():
        global belief
        measurements = np.array([((*b[0]), b[1]) for b in belief])
        plt.scatter(measurements[:,0], measurements[:,1], c=measurements[:,2])
        plt.pause(0.01)


if __name__ == "__main__":

    start_mavlink(connection_url="udp://:14552")
    drone = mavsdk_connect(host="127.0.0.1")

    belief = []

    #fill in the space with 2D gaussian 
    N_CELLS = 100 # 1 for cell equiv. to 1 m

    x, y = np.meshgrid(np.linspace(-1,1,N_CELLS), np.linspace(-1,1,N_CELLS)) #100x100 cells
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    truth = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

    # define space
    sp = Space(origin=(37.331553, -121.882767))

    # generate trajectory
    traj = []
    step = 5 # [grid step of 10m]
    for x in range(0,N_CELLS,step):
        traj.append((x,0))
        traj.append((x,N_CELLS)) # that's assuming N_CELLS is equal to search space size
        traj.append((x+step/2,N_CELLS))
        traj.append((x+step/2,0))

    # get trajectory in global frame
    traj_global = list(map(lambda p: sp.xy2latlon(*p), traj))

    # do the drone stuff
    asyncio.ensure_future(run(traj_global))
    asyncio.ensure_future(print_mission_progress())
    asyncio.ensure_future(print_battery(drone))
    asyncio.ensure_future(measure(drone))
    asyncio.get_event_loop().run_until_complete(observe_is_in_air())


    # convert belief space to numpy array
    measurements = np.array([((*b[0]), b[1]) for b in belief])

    plt.scatter(measurements[:,0], measurements[:,1], c=measurements[:,2], s=1)

    # 3D plotting would be:
    # >>> from mpl_toolkits.mplot3d import Axes3D
    # >>> fig = plt.figure()
    # >>> ax = fig.add_subplot(111, projection='3d')
    # >>> ax.scatter(measurements[:,0], measurements[:,1], measurements[:,2], c=measurements[:,2]

    plt.show()
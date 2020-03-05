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


class Drone:

    def __init__(self, url="udp://:14552"):
        self._mav = start_mavlink(connection_url=url)
        self._drone = mavsdk_connect(host="127.0.0.1")

    async def run_scan(self, traj):

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
                                        

        await self._drone.mission.set_return_to_launch_after_mission(True)

        print("-- Uploading mission")
        await self._drone.mission.upload_mission(mission_items)

        print("-- Arming")
        await self._drone.action.arm()

        print("-- Starting mission")
        await self._drone.mission.start_mission()


    async def print_mission_progress(self):
        async for mission_progress in self._drone.mission.mission_progress():
            print(f"Mission progress: {mission_progress.current_item_index}/{mission_progress.mission_count}")


    async def observe_is_in_air(self):
        """ Monitors whether the drone is flying or not and
        returns after landing """

        was_in_air = False

        async for is_in_air in self._drone.telemetry.in_air():
            if is_in_air:
                was_in_air = is_in_air

            if was_in_air and not is_in_air:
                await asyncio.get_event_loop().shutdown_asyncgens()
                return

    async def print_pos(self, belief):
        async for pos in self._drone.telemetry.position():
            print(pos)

    async def print_battery(self):
        async for battery in self._drone.telemetry.battery():
            print(f"Battery: {battery.remaining_percent}")

    async def measure(self, belief, space_transform, ground_truth):
        async for position in self._drone.telemetry.position():
            # sample the truth in current position & append to belief
            coords = space_transform.latlon2xy(position.latitude_deg, position.longitude_deg)
            fire_value = ground_truth[int(coords[0]), int(coords[1])]
            print(fire_value)
            belief.append(coords, fire_value)

    async def plot_belief(self):
        async for pos in self._drone.telemetry.position():
            global belief
            measurements = np.array([((*b[0]), b[1]) for b in belief])
            plt.scatter(measurements[:,0], measurements[:,1], c=measurements[:,2])
            plt.pause(0.01)

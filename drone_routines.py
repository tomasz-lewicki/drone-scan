import asyncio

from mavsdk import start_mavlink
from mavsdk import connect as mavsdk_connect
from mavsdk import (MissionItem)


class Drone:

    def __init__(self, url="udp://:14552"):
        self._mav = start_mavlink(connection_url=url)
        self._drone = mavsdk_connect(host="127.0.0.1")

    async def run_scan(self, traj, alt=5, speed=10):

        mission_items = []

        for p in traj:
            mission_items.append(MissionItem(p[0],
                                            p[1],
                                            alt,
                                            speed,
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
            try:
                fire_value = ground_truth[int(coords[0]), int(coords[1])]
            except IndexError:
                fire_value = -1
            # print(f'found {fire_value} fire')
            belief.append((coords, fire_value))


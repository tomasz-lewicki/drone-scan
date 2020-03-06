import numpy as np
from geopy import Point

import geopy, geopy.distance
import math

import math
from math import cos


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

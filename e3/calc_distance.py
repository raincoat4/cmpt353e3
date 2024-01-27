from xml.dom import minidom
import sys
import pandas as pd
from numpy import radians, sin, cos, sqrt, arctan2, pi

#haversine and deg2rad functions adapted to python from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206 
def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(deg2rad(lat1))*cos(deg2rad(lat2))*sin(dLon/2)**2
    c = 2000*arctan2(sqrt(a), sqrt(1-a))
    d = r*c
    return d

def deg2rad(deg):
    return deg*pi/180

def distance(points):
    dLat = points['lat'].diff().shift(-1)
    dLon = points['lon'].diff().shift(-1)

    distance_list = haversine(points['lat'], points['lon'], points['lat'].shift(-1), points['lon'].shift(-1))
    
    return distance_list.sum()

walk1 = sys.argv[1]

data = {'lat': [], 'lon': []}

doc = minidom.parse(walk1)
trkpts = doc.getElementsByTagName('trkpt')

for trkpt in trkpts:
    lat = trkpt.getAttribute('lat')
    lon = trkpt.getAttribute('lon')
    data['lat'].append(float(lat))
    data['lon'].append(float(lon))

df = pd.DataFrame(data)
print('Unfiltered distance: %0.2f' % distance(df))


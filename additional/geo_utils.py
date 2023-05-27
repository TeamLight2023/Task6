from urllib.request import urlopen, Request, ProxyHandler, build_opener, install_opener
import requests
import overpy
from decimal import Decimal
from yandex_geocoder import Client
import geopy
from geopy.geocoders import Nominatim, GoogleV3, Yandex
import time
import xml.etree.ElementTree as ET
import ssl
from datetime import datetime
import json
from math import sin, cos, sqrt, atan2, radians
import numpy as np

ahml_http_proxy = "http://ivan.zaputlyaev:q1234Asdf@proxy.ahml1.ru:3128"
ahml_https_proxy = "https://ivan.zaputlyaev:q1234Asdf@proxy.ahml1.ru:3128"

ahml_proxyDict = {
    "http": ahml_http_proxy,
    "https": ahml_https_proxy,
}

yandex_api_key = '392ceaff-76cf-4ee2-b72b-761db7b62d4d'
here_api_key = '2xFtllLdxrTP3CduQpKzqSF9410ksNB_Ozlmbr4xtOs'


# #################################
# Connectors to geo services
# #################################

class HereConnector(object):
    def __init__(self, api_key):
        '''
        #in
        api_key: str
        
        #out
        None
        '''

        self.api_key = api_key
        self.base_url = "https://geocode.search.hereapi.com/v1/geocode"

    def address_to_coordinates(self, address, proxyDict=None, delay=None):
        '''
        #in
        address: str
        proxyDict: dict
        delay: int
        
        #out
        tuple (float, float)
        '''

        if not (delay is None):
            time.sleep(delay)

        params = {'apikey': self.api_key, 'q': address}
        response = requests.get(url=self.base_url, params=params, proxies=proxyDict)

        data = response.json()
        latitude = data['items'][0]['position']['lat']
        longitude = data['items'][0]['position']['lng']

        return latitude, longitude

    def coordinates_to_address(self, coordinates, proxyDict=None, delay=None):
        '''
        #in
        coordinates: tuple (latitude: float, longitude: float)
        proxyDict: dict
        delay: int
        
        #out
        str
        '''

        pass


class OverPassConnector(object):
    def __init__(self):
        self.baseurl = 'https://www.openstreetmap.org/api/0.6/node/{}/{}'
        pass

    def generate_around_query_for_point(self, coordinates, radius, tags):
        '''
        #in
        coordinates: tuple (latitude: float, longitude: float)
        radius: int
        tags: list of str
        
        #out
        overpass object
        '''

        latitude, longitude = coordinates
        s = ''
        for tag in tags:
            s += "node[%s](around:%i,%s,%s);" % (tag, radius, latitude, longitude)
        s = '(' + s + ');out meta;'

        return s

    def generate_around_query_for_point2(self, coordinates, radius, tags):
        '''
        #in
        coordinates: tuple (latitude: float, longitude: float)
        radius: int
        tags: list of str
        
        #out
        overpass object
        '''

        latitude, longitude = coordinates
        s = ''
        for tag in tags:
            s += 'node[%s](around:%i,%s,%s)(older than:"2011-08-01T00:00:00Z");' % (tag, radius, latitude, longitude)
        s = '(' + s + ');out meta;'

        return s

    def generate_around_query_for_points(self, coordinates, radius, tags):
        '''
        #in
        coordinates: list of tuples (latitude: float, longitude: float)
        radius: int
        tags: list of str
        
        #out
        overpass object
        '''

        s = ''
        for lat, lon in coordinates:
            for tag in tags:
                s += 'node[%s](around:%i,%s,%s);' % (tag, radius, lat, lon)
        s = '(' + s + ');out bb;'

        return s

    def count_infrastructure_around_address(self, address, radius, tags, api_key, proxyDict=None, delay=None):
        '''
        #in
        address: str
        radius: int
        tags: list of str
        api_key: str
        proxyDict: dict
        delay: int
        
        #out
        int
        '''

        if not (delay is None):
            time.sleep(delay)

        here_con = HereConnector(api_key)
        lat, lon = here_con.address_to_coordinates(address, proxyDict)
        q = self.generate_around_query_for_points([(lat, lon)], radius, tags)
        # return q
        overpass_api = overpy.Overpass()
        result = overpass_api.query(q, proxyDict)

        return len(result.nodes)

    def count_infrastructure_around_point(self, coordinates, radius, tags, proxyDict=None, delay=None):
        '''
        #in
        coordinates: tuple (float, float)
        radius: int
        tags: list of str
        proxyDict: dict
        delay: int
        
        #out
        int
        '''

        if not (delay is None):
            time.sleep(delay)

        lat, lon = coordinates
        q = self.generate_around_query_for_point((lat, lon), radius, tags)
        overpass_api = overpy.Overpass()
        # result = overpass_api.query(q, proxyDict)
        result = overpass_api.query(q)

        return (result.nodes)

        return len(result.nodes)

    def count_infrastructure_around_point2(self, coordinates, radius, tags, proxyDict=None, delay=None):
        '''
        #in
        coordinates: tuple (float, float)
        radius: int
        tags: list of str
        proxyDict: dict
        delay: int
        
        #out
        int
        '''

        if not (delay is None):
            time.sleep(delay)

        lat, lon = coordinates
        q = self.generate_around_query_for_point2((lat, lon), radius, tags)
        overpass_api = overpy.Overpass()
        # result = overpass_api.query(q, proxyDict)
        result = overpass_api.query(q)

        return (result.nodes)

        return len(result.nodes)

    def get_first_version_of_node(self, nodeid, originaltimestamp, version=1, proxyDict=None):
        url = self.baseurl.format(str(nodeid), str(version))
        context = ssl._create_unverified_context()
        response = urlopen(url=url, context=context).read()
        root = ET.fromstring(response)
        for child in root:
            if child.attrib['timestamp'] != None:
                date = datetime.strptime(child.attrib['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
                return date
        return originaltimestamp


class OSRMConnector():
    def __init__(self):
        self.baseurl_car = 'http://router.project-osrm.org/route/v1/car/{lon_1}%2C{lat_1}%3B{lon_2}%2C{lat_2}?overview=false'
        # self.baseurl_foot = 'https://routing.openstreetmap.de/routed-foot/route/v1/foot/37.579105,55.729574%3B37.609934,55.746257?overview=false'
        self.baseurl_foot = 'https://routing.openstreetmap.de/routed-foot/route/v1/foot/{lon_1}%2C{lat_1}%3B{lon_2}%2C{lat_2}?overview=false'
        pass

    def get_route_car(self, node_1, node_2):
        url = self.baseurl_car.format(lon_1=node_1[1], lat_1=node_1[0], lon_2=node_2[1], lat_2=node_2[0])
        # context = ssl._create_unverified_context()

        response = urlopen(url=url).read()
        # return response
        routes = json.loads(response)
        return routes['routes'][0]

    def get_route_foot(self, node_1, node_2):
        url = self.baseurl_foot.format(lon_1=node_1[1], lat_1=node_1[0], lon_2=node_2[1], lat_2=node_2[0])
        context = ssl._create_unverified_context()

        response = urlopen(url=url, context=context).read()
        # return response
        routes = json.loads(response)
        return routes['routes'][0]

    def get_foot_url(self, node_1, node_2):
        url = self.baseurl_foot.format(lon_1=node_1[1], lat_1=node_1[0], lon_2=node_2[1], lat_2=node_2[0])
        return url


def geodistanse(lats1, lons1, lats2, lons2):
    # approximate radius of earth in km
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lats1, lons1, lats2, lons2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

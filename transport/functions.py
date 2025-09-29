import requests
import numpy as np
from shapely.geometry import Polygon, Point
import random

def osrm_distance(lat1, lon1, lat2, lon2, profile="driving"):
    """
    Compute distance using OSRM public API.
    profile = "driving", "walking", or "cycling"
    """
    url = f"http://router.project-osrm.org/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}?overview=false"
    response = requests.get(url).json()

    # Distance in meters, duration in seconds
    distance_m = response["routes"][0]["distance"]
    #duration_s = response["routes"][0]["duration"]

    return distance_m / 1000#, duration_s / 60

def osrm_table(origins, destinations, profile="driving"):
    """
    Compute distance matrix between origins and destinations using OSRM.
    origins: array/list of (lat, lon)
    destinations: array/list of (lat, lon)
    Returns: np.array with distances in km
    """
    # Build coordinates string (lon,lat required by OSRM)
    all_coords = ";".join([f"{lon},{lat}" for lat, lon in origins + destinations])

    # sources = 0..len(origins)-1, destinations = len(origins)..end
    src_idx = ";".join(map(str, range(len(origins))))
    dst_idx = ";".join(map(str, range(len(origins), len(origins) + len(destinations))))

    url = (
        f"http://router.project-osrm.org/table/v1/{profile}/{all_coords}"
        f"?sources={src_idx}&destinations={dst_idx}&annotations=distance"
    )

    response = requests.get(url).json()
    distances = np.array(response["distances"]) / 1000  # convert m → km
    return distances

def random_point_in_polygon(polygon, n_points):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    while len(points) < n_points:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            points.append((random_point.x, random_point.y))
    return points


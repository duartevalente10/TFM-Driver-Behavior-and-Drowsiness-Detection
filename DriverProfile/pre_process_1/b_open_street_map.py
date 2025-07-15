import overpy
import time
from urllib.error import URLError


def is_over_speed_limit(lat, lon, current_speed, radius=30):
    """Get speed limits from OSM's Overpass API within
    a certain radius (in meters) of gps coordinates and
    checks if vehicle is over limit

    Args:
        lat (float): Latitude
        lon (float): Longitude
        current_speed (float): Current vehicle speed (km/h)
        radius (int, optional): Radius from the coordinates. Defaults to 30.

    Returns:
        bool: True if vehicle exceeded the speed limit
    """

    # with overpass api
    api = overpy.Overpass()

    #  - maxspeed comes in km/h.
    #  - highway: highway type.
    #  - lanes: The number of traffic lanes for general purpose traffic, also
    #           for buses and other specific classes of vehicle.
    #  - overtaking: Specifying sections of roads where overtaking is legally
    #                forbidden.

    # initialize roads list
    roads = []

    try:
        r = api.query(
            """
            [out:json];
            way(around:{},{},{})["maxspeed"];
            (._;>;);
            out body qt;
            """.format(radius, lat, lon)
        )

        # loop through every road (way) (Not in order!!)
        for way in r.ways:
            road = {}
            road["name"] = way.tags.get("name", "n/a")
            road["speed_limit"] = way.tags.get("maxspeed", "n/a")
            road["highway"] = way.tags.get("highway", "n/a")
            road["lanes"] = way.tags.get("lanes", "n/a")
            road["overtaking"] = way.tags.get("overtaking", "n/a")
            nodes = []
            for node in way.nodes:
                nodes.append((node.lat, node.lon))
            road["nodes"] = nodes
            roads.append(road)

        radius_step = 5
    except (
        overpy.exception.OverpassGatewayTimeout,
        overpy.exception.OverpassTooManyRequests,
        URLError
    ):
        radius_step = 0
        print('Waiting 2 seconds...')
        time.sleep(2)  # sleep 2 seconds

    # stop conditions
    if len(roads) > 0:  # got a road
        return current_speed > float(roads[0]['speed_limit'])
    elif radius > 50:  # exceeded the radius limit
        print('No speed limit found...')
        return False
    else:
        return is_over_speed_limit(
            lat, lon, current_speed, radius + radius_step
        )


if __name__ == "__main__":

    lat = 38.7546664  # 38.7673366
    lon = -9.1441218  # -9.1493435
    current_speed = 80.0
    radius = 50

    roads = is_over_speed_limit(lat, lon, current_speed, radius)
    print(roads)

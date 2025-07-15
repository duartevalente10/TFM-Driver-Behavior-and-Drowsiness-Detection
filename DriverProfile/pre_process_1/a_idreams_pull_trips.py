# -*- coding: utf-8 -*-
"""
idreams_pull_trips
------------------

Script to pull i-Dreams trips from the Cardio API.

:copyright: (c) 2021 by CardioID Technologies Lda.
:license: All rights reserved.
"""

# Imports
import argparse
import datetime
import json
import os
import pandas
import requests


#def get_token(index=0):
"""Get token from tokens file.

    Parameters
    ----------
    index : str
        Token index from the list of tokens.

    Returns
    -------
    res : str
        Token.

    """
    #tokens = open("", "r")
    #tokens = open("../private/tokens.json", "r")
    #data = json.load(tokens)
    #token = list(data)[index]['token']
    #tokens.close()
    #return token


# ----- API Token -----
#TOKEN = get_token()
TOKEN = ""


# Globals
SRV = 'https://api.cardio-id.com'
HDRS = {
    'Auth-Token': TOKEN,
    'Content-Type': 'application/json',
}


class APIError(Exception):
    """Class for API request exceptions."""


def get_request(url):
    """Do a GET request to an URL.

    Parameters
    ----------
    url : str
        Endpoint URL.

    Returns
    -------
    res : dict
        Response JSON object.

    """

    resp = requests.get(url, headers=HDRS)

    if not resp.ok:
        txt = 'GET request failed with HTTP status {}.'
        raise APIError(txt.format(resp.status_code))

    return resp.json()


def post_request(url, query):
    """Post a request to an URL with a JSON query.

    Parameters
    ----------
    url : str
        Endpoint URL.
    query : dict
        JSON object with query parameters.

    Returns
    -------
    res : dict
        Response JSON object.

    """

    resp = requests.post(url, headers=HDRS, json=query)

    if not resp.ok:
        txt = 'POST request failed with HTTP status {}.'
        raise APIError(txt.format(resp.status_code))

    return resp.json()


def get_vehicles():
    """Get vehicles from API.

    Returns
    -------
    vehicles : List[Dict]
        Retrieved vehicles.

    """

    url = SRV + '/idreams/vehicles'

    try:
        vehicles = get_request(url)
    except APIError as err:
        print('Failed to get vehicles from API: ', err)
        vehicles = []

    return vehicles


def get_vehicle_trips(vehicle, ti, tf):
    """Get trips for given vehicle.

    Parameters
    ----------
    vehicle : str
        Vehicle UUID.
    ti : datetime
        Query start date.
    tf = datetime
        Query end date.

    Returns
    -------
    trips : List[Dict]
        Retrieved trips.

    """

    url = SRV + '/idreams/vehicle/trips'

    query = {
        'uuid': vehicle,
        'date_from': ti.isoformat(),
        'date_to': tf.isoformat(),
    }

    try:
        trips = post_request(url, query)
    except APIError as err:
        print('Failed to get trips for {}:'.format(vehicle), err)
        trips = []

    return trips


def get_trip_data(trip):
    """Get data for given trip and store it to file.

    Parameters
    ----------
    trip : str
        Trip UUID.

    Returns
    -------
    data : Dict
        Retrieved trip data.

    """

    url = SRV + '/idreams/trip/data'

    query = {
        'uuid': trip,
    }

    try:
        data = post_request(url, query)
    except APIError as err:
        print('Failed to get data for trip {}:'.format(trip), err)
        data = {}

    return data


def unpack_tram_sim_data(data):
    """Unpack the JSON from the Tram simulator.

    Parameters
    ----------
    data : Dict
        Simulator trip data.

    Returns
    -------
    frame : pandas.DataFrame
        Unpacked data.

    """

    # get column names
    rows = []
    cols = set()
    for row in data.get('data', []):
        if isinstance(row, str):
            # row is a JSON string, unpack it
            aux = json.loads(row)
        else:
            # row is directly a JSON
            aux = row

        rows.append(aux)
        for k in aux:
            cols.add(k)

    # prepare output dict
    out = {
        'ts': data.get('ts', []),
    }

    for k in cols:
        out[k] = []

    # unpack dicts
    for row in rows:
        for k in cols:
            out[k].append(row.get(k, None))

    frame = pandas.DataFrame(out)

    return frame


def store_csv(path, dtype, data):
    """Store trip data to CSV file.

    Parameters
    ----------
    path : str
        Path to storage location.
    dtype : str
        Data type.
    data : Dict
        Data to store.

    """

    fpath = os.path.join(path, '{}.csv'.format(dtype))

    date_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'

    if dtype == 'iDreams_Tram_Sim':
        frame = unpack_tram_sim_data(data)
    else:
        frame = pandas.DataFrame(data)

    frame.to_csv(fpath, header=True, index=False, date_format=date_fmt)


def store_trip(path, info, data):
    """Store trip.

    Parameters
    ----------
    path : str
        Path to storage location.
    info : Dict
        Trip info.
    data : Dict
        Trip data.

    """

    ts = datetime.datetime.fromisoformat(info['trip_start'])

    d = ts.strftime('%Y_%m_%dT%H_%M_%S')
    t = info['uuid'][-5:]

    fpath = os.path.join(path, '{}__{}'.format(d, t))
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # store trip info
    with open(os.path.join(fpath, 'info.json'), 'w') as fid:
        json.dump(info, fid)

    # store trip data
    for dtype in data['data']:
        store_csv(fpath, dtype, data['data'][dtype])


def parse_input_date(txt):
    """Parse input date string.

    Parameters
    ----------
    txt : str
        Date string.

    Returns
    -------
    dt : datetime.datetime
        Parsed date.

    """

    dt = datetime.datetime.strptime(txt, '%Y-%m-%d')

    return dt


def main(ti_str, tf_str=None):
    """Main.

    Parameters
    ----------
    ti_str : str
        Query start date string.
    tf_str : Optional[str]
        Query end date string; if None, defaults to current time.

    """

    path = os.path.join(os.getcwd(), '../trips')

    # parse dates
    try:
        ti = parse_input_date(ti_str)
    except ValueError:
        print('Could not parse query start date, expected format is YYYY-MM-DD.')
        return

    if tf_str is None:
        tf = datetime.datetime.utcnow()
    else:
        try:
            tf = parse_input_date(tf_str)
        except ValueError:
            print('Could not parse query end date, expected format is YYYY-MM-DD.')
            return
        else:
            # change to end of day
            tf = tf.replace(hour=23, minute=59)

    print('Requesting trips from {} to {}.'.format(ti.isoformat(), tf.isoformat()))

    vehicles = get_vehicles()
    n = 0

    for v in vehicles:
        trips = get_vehicle_trips(v['uuid'], ti, tf)

        for t in trips:
            print(v['uuid'], t['uuid'])
            trip_data = get_trip_data(t['uuid'])
            store_trip(path, t, trip_data)
            n += 1

    print('Downloaded {} trips.'.format(n))


def cli():
    """Command Line Interface."""

    p = 'idreams_pull_trips.py'
    d = 'Pull i-Dreams trips from Cardio API.'
    e = 'Copyright (c) 2021 CardioID Technologies, Lda.'

    parser = argparse.ArgumentParser(prog=p, description=d, epilog=e)

    parser.add_argument(
        '-ti',
        metavar='YYYY-MM-DD',
        type=str,
        required=True,
        help='Trip query start date (e.g. "2021-04-21")',
    )

    parser.add_argument(
        '-tf',
        metavar='YYYY-MM-DD',
        type=str,
        required=False,
        help='Trip query end date (e.g. "2021-04-21"); if not provided, uses current date',
    )

    args = parser.parse_args()
    main(args.ti, args.tf)


if __name__ == '__main__':
    cli()

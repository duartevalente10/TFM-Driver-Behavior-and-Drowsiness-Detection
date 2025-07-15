# packages
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
# local
from .b_open_street_map import is_over_speed_limit
from .a_idreams_pull_trips import (
    parse_input_date,
    get_vehicles,
    get_vehicle_trips,
    get_trip_data
)


def read_all_trips_and_store_df(path, datasets_path, dataset_name):
    """
    Read all trips from folder and create dataset

    Args:
        path (str): Trips path
        datasets_path (str): Datasets path
        dataset_name (str): Dataset name
    """
    dataset = None
    for _, subdirectories, _ in os.walk(path):
        for subdirectory in subdirectories:
            print(subdirectory)
            folder = '../trips/{}/'.format(subdirectory)
            info = read_json_file(folder + 'info.json')
            event = construct_dataset(folder, info)
            if dataset is None:
                dataset = event
            else:
                dataset = pd.concat([dataset, event])
            print('\n')
    print(dataset)
    store_csv(datasets_path, dataset_name, dataset)


def pull_trips_and_store_df(datasets_path, dataset_name, ti_str, tf_str=None):
    """
    Pull trips from the Cardio-ID API and create dataset

    Args:
        datasets_path (str): Datasets path
        dataset_name (str): Dataset name
        ti_str (str): Query start date string
        tf_str (str): Query end date string; if None, defaults to current time
    """

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
    dataset = None

    for v in vehicles:
        trips = get_vehicle_trips(v['uuid'], ti, tf)
        for t in trips:
            print(v['uuid'], t['uuid'])
            trip_data = get_trip_data(t['uuid'])
            dtype_df = construct_dataset(trip_data['data'], t)
            if dataset is None:
                dataset = dtype_df
            else:
                dataset = pd.concat([dataset, dtype_df])
            n += 1
    store_csv(datasets_path, dataset_name, dataset)
    print('Downloaded {} trips.'.format(n))


def read_csv_file(file):
    """
    Read CSV file

    Args:
        file (str): File path name

    Returns:
        pandas.DataFrame: Dataframe table
    """
    try:
        data = pd.read_csv(file + '.csv')
        return data
    except FileNotFoundError:
        print('No such file:', file + '.csv')
    except pd.errors.EmptyDataError:
        print('No columns to parse from file:', file + '.csv')


def store_csv(path, dataset_name, dataset):
    """
    Store trips dataset to CSV file.

    Args:
        path (str): Path to storage location
        dataset_name (str): Dataset name
        dataset (pandas.DataFrame): Dataset to store.
    """
    if dataset is not None:
        fpath = os.path.join(path, '{}.csv'.format(dataset_name))
        date_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
        dataset.to_csv(fpath, header=True, index=False, date_format=date_fmt)


def read_json_file(file):
    """
    Read json file

    Args:
        file (str): File name

    Returns:
        dict: Data in json file
    """
    try:
        with open(file, 'r') as f:
            data = json.loads(f.read())
        return data
    except FileNotFoundError:
        print('No such file:', file)


def join_datasets(df1, df2):
    """
    Concatenate datasets by rows

    Args:
        df1 (pandas.DataFrame): Dataset 1
        df2 (pandas.DataFrame): Dataset 2

    Returns:
        pandas.DataFrame: Dataset DataFrame
    """
    return pd.concat([df1, df2])


def construct_dataset(data, info):
    """
    Construct dataset with all the events available for each trip

    Args:
        data (dict): Trip data
        info (dict): Trip default info

    Returns:
        pandas.DataFrame: Dataset DataFrame
    """
    dtypes = info['dtypes']
    del info['dtypes']
    del info['uuid']
    df = pd.DataFrame([info])
    not_events = [
        'ECG', 'IBI', 'CAN', 'DriverChange', 'oSeven_Trip', 'ME_Tamper_Map',
        'iDreams_Driver_Map', 'Driver_Map',
        'Drowsiness',  # Drowsiness Map is enough
        'LOD_Event',  # Hands On Detection Map is enough
        'DrivingEvents',  # Driving Events Map is enough
        'iDreams_Fatigue',  # Fatigue Map is enough
        'iDreams_Headway',  # Headway Map is enough
        'iDreams_Overtaking',  # Overtaking Map is enough
        'iDreams_Speeding',  # Speeding Map is enough
    ]
    events = np.setdiff1d(dtypes, not_events)

    for e in events:
        # ev = read_csv_file(folder + e)
        ev = pd.DataFrame(data[e])
        if e == 'LOD_Event_Map':
            # TODO: Confirmar se se a mao ficar no volante se continuam a
            # receber sinais ou é só no 1º contato
            df = edit_lod_event(ev, df)
        elif e == 'Drowsiness_Map':
            # TODO: Confirmar quais sao os niveis de drowsiness e pedir um
            # ficheiro de exemplo
            df = edit_drowsiness_events(ev, df)
        elif e == 'DrivingEvents_Map':
            df = edit_driving_events(ev, df)
        elif e == 'Distraction_Map':
            # TODO: Pedir um ficheiro de exmplo para confirmar se está bem
            # implementado
            df = edit_distraction(ev, df)
        elif e == 'Ignition':
            # TODO: Verificar se está bem calculado
            # Pedir um ficheiro de exmplo para confirmar se está bem
            df = edit_ignition_events(ev, df)
        elif e == 'ME_AWS':
            df = edit_me_aws_events(ev, df)
        elif e == 'ME_Car':
            # Includes the GPS (to get OpenStreetMapp overspeeding events)
            df = edit_me_car_events(ev, df, False)
        elif e == 'ME_FCW_Map':
            df = edit_me_fcw_map(ev, df)
        elif e == 'ME_HMW_Map':
            df = edit_me_hmw_map(ev, df)
        elif e == 'ME_LDW_Map':
            df = edit_me_ldw_map(ev, df)
        elif e == 'ME_PCW_Map':
            df = edit_me_pcw_map(ev, df)
        elif e == 'iDreams_Fatigue_Map':
            df = edit_idreams_fatigue(ev, df)
        elif e == 'iDreams_Headway_Map':
            df = edit_idreams_headway(ev, df)
        elif e == 'iDreams_Overtaking_Map':
            df = edit_idreams_overtaking(ev, df)
        elif e == 'iDreams_Speeding_Map':
            df = edit_idreams_speeding(ev, df)
        elif e == 'GPS':
            df = edit_gps_event(ev, df)
        # elif e == 'DriverChange':
        #     driver = None
        #     if ev is not None:
        #         driver = ev['uuid'].iloc[0]
        #         print('driver:', driver)
        #     df['driver'] = driver
    return df


def edit_gps_event(ev, df):
    """
    Event indicating the trip start coordinates.

    Generated features:
        lat (int): Latitude
        lon (int): Longitude

    Args:
        ev (pandas.DataFrame): GPS DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    lat = lon = None
    if ev is not None:
        lat = ev['lat'].iloc[0]
        lon = ev['lon'].iloc[0]
    # update dataframe
    df['lat'] = lat
    df['lon'] = lon
    return df


def edit_lod_event(ev, df):
    """
    Event indicating the detection of the driver's hands on the steering wheel.

    Generated features:
        n_lod_0 (int): Number of no hands events
        n_lod_1 (int): Number of left hand events
        n_lod_2 (int): Number of right hand events
        n_lod_3 (int): Number of both hand events

    Args:
        ev (pandas.DataFrame): LOD_Event_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_lod_0 = n_lod_1 = n_lod_2 = n_lod_3 = None
    if ev is not None:
        lod_indexes = ev.ne(ev.shift()).filter(['type']) \
            .apply(lambda x: x.index[x].tolist())
        lod_data = ev.iloc[lod_indexes.values.ravel()]['type']
        # number of no hands 0 events
        n_lod_0 = sum(lod_data == 0)
        # number of left hand 1 events
        n_lod_1 = sum(lod_data == 1)
        # number of right hand 2 events
        n_lod_2 = sum(lod_data == 2)
        # number of both hands 3 events
        n_lod_3 = sum(lod_data == 3)
    # update dataframe
    df['n_lod_0'] = n_lod_0
    df['n_lod_1'] = n_lod_1
    df['n_lod_2'] = n_lod_2
    df['n_lod_3'] = n_lod_3
    return df


def edit_drowsiness_events(ev, df):
    """
    Event indicating driver drowsiness level (based on KSS).

    Generated features:
        TODO: O que significam os níveis
        n_drowsiness_0 (int): Number of drowsiness events level 0
        n_drowsiness_1 (int): Number of drowsiness events level 1
        n_drowsiness_2 (int): Number of drowsiness events level 2
        n_drowsiness_3 (int): Number of drowsiness events level 3

    Args:
        ev (pandas.DataFrame): Drowsiness_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    drowsiness_0 = drowsiness_1 = drowsiness_2 = drowsiness_3 = None
    if ev is not None:
        # number of drowsiness events level 0
        drowsiness_0 = sum(ev['level'] == 0)
        # number of drowsiness events level 1
        drowsiness_1 = sum(ev['level'] == 1)
        # number of drowsiness events level 2
        drowsiness_2 = sum(ev['level'] == 2)
        # number of drowsiness events level 3
        drowsiness_3 = sum(ev['level'] == 3)
    # update dataframe
    df['n_drowsiness_0'] = drowsiness_0
    df['n_drowsiness_1'] = drowsiness_1
    df['n_drowsiness_2'] = drowsiness_2
    df['n_drowsiness_3'] = drowsiness_3


def edit_driving_events(ev, df):
    """
    Events indicating occurrence of harsh acceleration, harsh braking,
    and harsh cornering behaviours.

    Generated features:
        TODO: Acrescentar o tempo o tempo de cada evento ??
        n_ha (int): Number of harsh acceleration events
        n_ha_l (int): Number of harsh acceleration events with low severity
        n_ha_m (int): Number of harsh acceleration events with medium severity
        n_ha_h (int): Number of harsh acceleration events with high severity
        n_hb (int): Number of harsh braking events
        n_hb_l (int): Number of harsh braking events with low severity
        n_hb_m (int): Number of harsh braking events with medium severity
        n_hb_h (in: Number of harsh cornering events
        n_hc_l (int): Number of harsh braking events with high severity
        n_hc (int)t): Number of harsh cornering events with low severity
        n_hc_m (int): Number of harsh cornering events with medium severity
        n_hc_h (int): Number of harsh cornering events with high severity

    Args:
        ev (pandas.DataFrame): DrivingEvents_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    ha = hb = hc = None
    ha_l = ha_m = ha_h = None
    hb_l = hb_m = hb_h = None
    hc_l = hc_m = hc_h = None
    if ev is not None:
        # number of harsh acceleration events (low, medium and high)
        ha = sum(ev['evt'] == 'ha')
        ha_l = sum((ev['evt'] == 'ha') & (ev['lvl'] == 'M'))
        ha_m = sum((ev['evt'] == 'ha') & (ev['lvl'] == 'L'))
        ha_h = sum((ev['evt'] == 'ha') & (ev['lvl'] == 'H'))
        # number of harsh braking events (low, medium and high)
        hb = sum(ev['evt'] == 'hb')
        hb_l = sum((ev['evt'] == 'hb') & (ev['lvl'] == 'M'))
        hb_m = sum((ev['evt'] == 'hb') & (ev['lvl'] == 'L'))
        hb_h = sum((ev['evt'] == 'hb') & (ev['lvl'] == 'H'))
        # number of harsh cornering events (low, medium and high)
        hc = sum(ev['evt'] == 'hc')
        hc_l = sum((ev['evt'] == 'hc') & (ev['lvl'] == 'M'))
        hc_m = sum((ev['evt'] == 'hc') & (ev['lvl'] == 'L'))
        hc_h = sum((ev['evt'] == 'hc') & (ev['lvl'] == 'H'))
    # update dataframe
    df['n_ha'] = ha
    df['n_ha_l'] = ha_l
    df['n_ha_m'] = ha_m
    df['n_ha_h'] = ha_h
    df['n_hb'] = hb
    df['n_hb_l'] = hb_l
    df['n_hb_m'] = hb_m
    df['n_hb_h'] = hb_h
    df['n_hc'] = hc
    df['n_hc_l'] = hc_l
    df['n_hc_m'] = hc_m
    df['n_hc_h'] = hc_h
    return df


def edit_distraction(ev, df):
    """
    Real-time mobile phone use events from the driver app.

    Generated features:
        distraction_time (float): Time spent distracted (seconds)
        n_distractions (int): Number of distraction events

    Args:
        ev (pandas.DataFrame): Distraction_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_distractions = distraction_time = None
    if ev is not None:
        # time spent distracted during trip
        event_indexes = ev.ne(ev.shift()).filter(['event']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(lambda row: row - 1 if row.name % 2 == 0 else row, axis=1)
        event_data = ev.iloc[event_indexes.values.ravel(), 0]
        event_date = pd.to_datetime(event_data, format='%Y-%m-%d')
        event_diff = event_date.diff(periods=1)[1:len(event_date) + 1:2]
        distraction_time = event_diff.sum().total_seconds()
        # number of distraction events
        n_distractions = len(event_diff)

    # update dataframe
    df['distraction_time'] = distraction_time
    df['n_distractions'] = n_distractions
    return df


def edit_ignition_events(ev, df):
    """
    Ignition events.

    Generated features:
        n_ignition_on (int): Number of ignition ON events
        n_ignition_off (int): Number of ignition OFF events

    Args:
        ev (pandas.DataFrame): Ignition DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    on = off = None
    if ev is not None:
        # number of times ignition is ON
        on = sum(ev['evt'] == 'ON')
        # number of times ignition is OFF
        off = sum(ev['evt'] == 'OFF')
    # update dataframe
    df['n_ignition_on'] = on
    df['n_ignition_off'] = off
    return df


def edit_me_aws_events(ev, df):
    """
    Gives information about the safety and warning state of the Mobileye
    system.

    Generated features:
        fcw_time (float): Time forward collision warning was active (seconds)
        hmw_time (float): Time headway monitoring was active (seconds)
        ldw_time (float): Time lane departure warning was active (seconds)
        pcw_time (float): Time pedestrian collision warning was
                          active (seconds)
        n_pedestrian_dz (int): Number times a pedestrian is detected in danger
                               zone
        light_mode (int): Trip Lighting condition (Most frequent)
        n_tsr_level (int): Number of times the speed limit was exceeded
        n_tsr_level_0 (int): Number of times the speed limit was not exceeded
        n_tsr_level_1 (int): Number of times 0-5 units over speed limit
        n_tsr_level_2 (int): Number of times 5-10 units over speed limit
        n_tsr_level_3 (int): Number of times 10-15 units over speed limit
        n_tsr_level_4 (int): Number of times 15-20 units over speed limit
        n_tsr_level_5 (int): Number of times 20-25 units over speed limit
        n_tsr_level_6 (int): Number of times 25-30 units over speed limit
        n_tsr_level_7 (int): Number of times 30+ units over speed limit
        zero_speed_time (float): Time the vehicle was stopped (seconds)
        n_zero_speed (int): Number of times the vehicle stopped
                            (first time ignition ON does not count)

    Args:
        ev (pandas.DataFrame): ME_AWS DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    fcw_time = hmw_time = ldw_time = pcw_time = zero_speed_time = None
    n_pedestrian_dz = n_tsr_level = n_tsr_level_0 = n_tsr_level_1 = None
    n_tsr_level_2 = n_tsr_level_3 = n_tsr_level_4 = n_tsr_level_5 = None
    n_tsr_level_6 = n_tsr_level_7 = n_zero_speed = None
    light_mode = None
    if ev is not None:
        # Para o fcw:
        # 1 - Calcular o numero de eventos fcw através do ME_FCW_Map
        # 2 - Calcular o tempo total (ou percentagem de tempo da trip) de fcw
        # na trip através do ME_AWS

        # fcw total time in trip
        fcw_indexes = ev.ne(ev.shift()).filter(['fcw']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        fcw_data = ev.iloc[fcw_indexes.values.ravel(), 0]
        fcw_date = pd.to_datetime(fcw_data, format='%Y-%m-%d')
        fcw_diff = fcw_date.diff(periods=1)[1:len(fcw_date) + 1:2]
        fcw_time = fcw_diff.sum().total_seconds()

        # Para o hw:
        # 1 - Calcular o numero de eventos hmw através do ME_HMW_Map
        # (hw_level = 2)
        # 2 - Calcular o tempo total (ou percentagem de tempo da trip) de hwmw
        # na trip através do ME_AWS

        # hmw total time in trip
        hmw_indexes = ev.ne(ev.shift()).filter(['hmw']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        hmw_data = ev.iloc[hmw_indexes.values.ravel(), 0]
        hmw_date = pd.to_datetime(hmw_data, format='%Y-%m-%d')
        hmw_diff = hmw_date.diff(periods=1)[1:len(hmw_date) + 1:2]
        hmw_time = hmw_diff.sum().total_seconds()

        # Para o ldw:
        # 1 - Calcular o numero de eventos ldw através do ME_LDW_Map
        # 2 - Calcular o numero de eventos ldw LEFT
        # através do ME_LDW_Map
        # 3 - Calcular o numero de eventos ldw RIGHT
        # através do ME_LDW_Map
        # 4 - Calcular o tempo total (ou percentagem de tempo da trip) de ldw
        # na trip através do ME_AWS

        # ldw total time in trip
        ldw_indexes = ev.ne(ev.shift()).filter(['ldw']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        ldw_data = ev.iloc[ldw_indexes.values.ravel(), 0]
        ldw_date = pd.to_datetime(ldw_data, format='%Y-%m-%d')
        ldw_diff = ldw_date.diff(periods=1)[1:len(ldw_date) + 1:2]
        ldw_time = ldw_diff.sum().total_seconds()

        # Para o pcw:
        # 1 - Calcular o numero de eventos pcw através do ME_PCW_Map
        # 2 - Calcular o tempo total (ou percentagem de tempo da trip) de hwmw
        # na trip através do ME_AWS

        # pcw total time in trip
        pcw_indexes = ev.ne(ev.shift()).filter(['pcw']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        pcw_data = ev.iloc[pcw_indexes.values.ravel(), 0]
        pcw_date = pd.to_datetime(pcw_data, format='%Y-%m-%d')
        pcw_diff = pcw_date.diff(periods=1)[1:len(pcw_date) + 1:2]
        pcw_time = pcw_diff.sum().total_seconds()

        # Para o pedestrian_dz:
        # 1 - Calcular o quantas vezes um pedestrian é detetado
        # através do ME_AWS

        pedestrian_dz_indexes = ev.ne(ev.shift()).filter(['pedestrian_dz']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        pedestrian_dz_data = ev.iloc[pedestrian_dz_indexes.values.ravel()]
        n_pedestrian_dz = len(
            pedestrian_dz_data[1:len(pedestrian_dz_data) + 1:2]
        )

        # Para o time_indicator:
        # 1 - Calcular a moda da hora do dia que mais acontece na trip pq
        # existem algumas falhas às vezes passa de dia para dusk e dps para dia
        # através do ME_AWS
        vals, counts = np.unique(ev['time_indicator'], return_counts=True)
        index = np.argmax(counts)
        light_mode = vals[index]
        print(light_mode)

        # Para o tsr:
        # 1 - Calcular o numero de vezes que ultrapassou o speed limit
        # através do ME_AWS
        # 2 - Calcular quantas vezes ultrapassou cada nivel
        # através do ME_AWS (nivel zero -> nao ultrapassou)
        # TODO: Questão: Os valores do tsr level estão > 0 mesmo com o tsr_on

        tsr_indexes = ev.ne(ev.shift()).filter(['tsr_level']) \
            .apply(lambda x: x.index[x].tolist())
        tsr_data = ev.iloc[tsr_indexes.values.ravel()]['tsr_level']

        n_tsr_level_0 = sum(tsr_data == 0)  # 21
        n_tsr_level_1 = sum(tsr_data == 1)
        n_tsr_level_2 = sum(tsr_data == 2)
        n_tsr_level_3 = sum(tsr_data == 3)
        n_tsr_level_4 = sum(tsr_data == 4)
        n_tsr_level_5 = sum(tsr_data == 5)
        n_tsr_level_6 = sum(tsr_data == 6)
        n_tsr_level_7 = sum(tsr_data == 7)
        # o numero de vezes que ultrapassou (o 0 nao conta)
        n_tsr_level = len(tsr_data[tsr_data != 0])  # 111

        # Para o zero_speed:
        # 1 - Calcular o numero de vezes que o veiculo parou através do ME_AWS
        # (so conta eventos de false para true) e não conta a paragem inicial
        # (inicio de marcha)
        # 2 - Calcular o tempo de viagem que o veiculo esteve parado

        zero_speed_indexes = ev.ne(ev.shift()).filter(['zero_speed']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 != 0 else row, axis=1
            )[1:]
        zero_speed_data = ev.iloc[zero_speed_indexes.values.ravel(), 0]
        zero_speed_date = pd.to_datetime(zero_speed_data, format='%Y-%m-%d')
        zero_speed_diff = zero_speed_date \
            .diff(periods=1)[1:len(zero_speed_date) + 1:2]
        zero_speed_time = zero_speed_diff.sum().total_seconds()
        n_zero_speed = len(zero_speed_diff)

    # update dataframe
    df['fcw_time'] = fcw_time
    df['hmw_time'] = hmw_time
    df['ldw_time'] = ldw_time
    df['pcw_time'] = pcw_time
    df['n_pedestrian_dz'] = n_pedestrian_dz
    df['light_mode'] = light_mode
    df['n_tsr_level'] = n_tsr_level
    df['n_tsr_level_0'] = n_tsr_level_0
    df['n_tsr_level_1'] = n_tsr_level_1
    df['n_tsr_level_2'] = n_tsr_level_2
    df['n_tsr_level_3'] = n_tsr_level_3
    df['n_tsr_level_4'] = n_tsr_level_4
    df['n_tsr_level_5'] = n_tsr_level_5
    df['n_tsr_level_6'] = n_tsr_level_6
    df['n_tsr_level_7'] = n_tsr_level_7
    df['zero_speed_time'] = zero_speed_time
    df['n_zero_speed'] = n_zero_speed
    return df


def edit_me_car_events(ev, df, osm_speed=True, gps_folder=''):
    """
    Gives information about the about the car parameters needed for the
    Mobileye system.

    Generated features:
        TODO: Time for each event ???
        n_high_beam (int): Number of times high beam is ON
        n_low_beam (int): Number of times low beam is ON
        n_wipers (int): Number of times wipers are ON
        n_signal_right (int): Number of times right turn signal is ON
        n_signal_left (int): Number of times left turn signal is ON
        n_brakes (int): Number of times breaks are ON
        speed (int): Mean Speed (km/h)
        over_speed_limit (int): Number of times over speed limit
                                (with openstreetmap)

    Args:
        ev (pandas.DataFrame): ME_Car DataFrame
        df (pandas.DataFrame): Dataset DataFrame
        osm_speed (bool, optional): Allow OSM overspeed events calculation.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: Dataset updated
    """
    high_beam = low_beam = wipers = signal_right = signal_left = None
    brakes = speed = None
    if ev is not None:
        # number of times high beam is ON
        high_beam_indexes = ev.ne(ev.shift()).filter(['high_beam']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        high_beam_data = ev.iloc[high_beam_indexes.values.ravel()]
        high_beam = len(high_beam_data[1:len(high_beam_data) + 1:2])
        # number of times low beam is ON
        low_beam_indexes = ev.ne(ev.shift()).filter(['low_beam']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        low_beam_data = ev.iloc[low_beam_indexes.values.ravel()]
        low_beam = len(low_beam_data[1:len(low_beam_data) + 1:2])
        # number of times wipers are ON
        wipers_indexes = ev.ne(ev.shift()).filter(['wipers']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        wipers_data = ev.iloc[wipers_indexes.values.ravel()]
        wipers = len(wipers_data[1:len(wipers_data) + 1:2])
        # number of times right turn signal is ON
        signal_right_indexes = ev.ne(ev.shift()).filter(['signal_right']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        signal_right_data = ev.iloc[signal_right_indexes.values.ravel()]
        signal_right = len(signal_right_data[1:len(signal_right_data) + 1:2])
        # number of times left turn signal is ON
        signal_left_indexes = ev.ne(ev.shift()).filter(['signal_left']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        signal_left_data = ev.iloc[signal_left_indexes.values.ravel()]
        signal_left = len(signal_left_data[1:len(signal_left_data) + 1:2])
        # number of times breaks are ON
        brakes = sum(ev['brakes'])
        brakes_indexes = ev.ne(ev.shift()).filter(['brakes']) \
            .apply(lambda x: x.index[x].tolist()) \
            .apply(
                lambda row: row - 1 if row.name % 2 == 0 else row, axis=1
            )[1:]
        brakes_data = ev.iloc[brakes_indexes.values.ravel()]
        brakes = len(brakes_data[1:len(brakes_data) + 1:2])
        # speed mean
        speed = np.mean(ev['speed'])

        # GPS overspeeding events
        overspeed_df = ev[['ts', 'speed']].copy()  # [100:105]
        # print('ME_Car', overspeed_df)

        # convert columns to appropriate type
        overspeed_df['ts'] = pd.to_datetime(overspeed_df['ts'])
        overspeed_df['speed'] = pd.to_numeric(overspeed_df['speed'])

        # line plot for gps
        # ax = overspeed_df.plot(
        #     kind='line',
        #     x='ts',
        #     y='speed',
        #     color='green',
        #     label='Speed'
        # )

        # read GPS table
        n_over_limit = None
        gps_ev = read_csv_file(gps_folder + 'GPS')

        if gps_ev is not None and osm_speed:
            gps_ts_speed = gps_ev[['ts']]  # [14:17]
            gps_ts_speed['speed'] = None

            # convert columns to appropriate type
            gps_ts_speed['ts'] = pd.to_datetime(gps_ts_speed['ts'])
            gps_ts_speed['speed'] = pd.to_numeric(gps_ts_speed['speed'])

            # add GPS (ts, speed=None) to the ME_Car (ts, speed) events
            overspeed_df = overspeed_df.append(gps_ts_speed, ignore_index=True)

            # sort rows by timestamp column and reset indexes
            # drop=True makes sure old indexes are not added as new column
            overspeed_df = overspeed_df.sort_values(by='ts') \
                .reset_index(drop=True)
            # print('Sorted', overspeed_df)

            # get the rows that need to calculate the speed
            speed_rows = overspeed_df[overspeed_df['speed'].isnull()]

            # # convert ts to only seconds since trip started
            # position = overspeed_df.columns.get_loc('ts')
            # overspeed_df['ts'] = overspeed_df.iloc[1:, position] - \
            #     overspeed_df.iat[0, position]
            # overspeed_df['ts'] = overspeed_df['ts'].dt.total_seconds()

            # make timestamp column the index (for the interpolate function)
            overspeed_df.index = overspeed_df['ts']
            del overspeed_df['ts']

            # linear interpolation
            interpolated = overspeed_df.interpolate(method='time') \
                .reset_index().loc[speed_rows.index, :]
            # print(interpolated)

            # if first speed is null -> GPS first ts is < ME_Car ts
            if interpolated.loc[interpolated.index[0], 'speed']:
                # fill with ME_Car first value
                interpolated.fillna(
                    interpolated[~interpolated['speed'].isnull()]
                    .head(1)['speed'].values[0],
                    inplace=True
                )

            # # line plot for me_car
            # interpolated.plot(
            #     kind='line',
            #     x='ts',
            #     y='speed',
            #     color='blue',
            #     label='Interpolated Speed',
            #     ax=ax
            # )
            # plt.title('GPS Speed Interpolation')
            # plt.savefig('images/gps_speed_interpolation')
            # plt.show()

            # count the number of speeding events
            # read lat and lon from GPS and speed from ME_Car
            speed_values = interpolated.loc[speed_rows.index, 'speed']
            # print('Indexes', speed_rows.index)
            # print('Speed values', speed_values)
            speed_limits = list(map(
                is_over_speed_limit, gps_ev['lat'], gps_ev['lon'], speed_values
            ))
            # print('Speed Limits:', speed_limits)
            n_over_limit = sum(speed_limits)
            # print('Number over limit:', n_over_limit)

    # update dataframe
    df['n_high_beam'] = high_beam
    df['n_low_beam'] = low_beam
    df['n_wipers'] = wipers
    df['n_signal_right'] = signal_right
    df['n_signal_left'] = signal_left
    df['n_brakes'] = brakes
    df['speed'] = speed
    df['over_speed_limit'] = n_over_limit
    return df


def edit_me_fcw_map(ev, df):
    """
    Computed geolocations of Mobileye FCW events.

    Generated features:
        n_fcw (int): Number of FCW events

    Args:
        ev (pandas.DataFrame): ME_FCW_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_fcw = None
    if ev is not None:
        # number of fcw events
        n_fcw = len(ev)
    # update dataframe
    df['n_fcw'] = n_fcw
    return df


def edit_me_hmw_map(ev, df):
    """
    Computed geolocations of Mobileye HMW events.

    Generated features:
        n_hmw (int): Number of HMW events

    Args:
        ev (pandas.DataFrame): ME_HMW_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_hmw = None
    if ev is not None:
        # number of hmw events
        n_hmw = len(ev)
    # update dataframe
    df['n_hmw'] = n_hmw
    return df


def edit_me_ldw_map(ev, df):
    """
    Computed geolocations of Mobileye LDW events.

    Generated features:
        n_ldw (int): Number of LDW events
        n_ldw_left (int): Number of LDW left events
        n_ldw_right (int): Number of LDW rigth events

    Args:
        ev (pandas.DataFrame): ME_LDW_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_ldw_left = n_ldw_right = n_ldw = None
    if ev is not None:
        # number of ldw left events
        n_ldw_left = sum(ev['type'] == 'L')
        # number of ldw right events
        n_ldw_right = sum(ev['type'] == 'R')
        # number of ldw left events
        n_ldw = len(ev)
    # update dataframe
    df['n_ldw'] = n_ldw
    df['n_ldw_left'] = n_ldw_left
    df['n_ldw_right'] = n_ldw_right
    return df


def edit_me_pcw_map(ev, df):
    """
    Computed geolocations of Mobileye PCW events.

    Generated features:
        n_pcw (int): Number of PCW events

    Args:
        ev (pandas.DataFrame): ME_PCW_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_pcw = None
    if ev is not None:
        # number of pcw events
        n_pcw = len(ev)
    # update dataframe
    df['n_pcw'] = n_pcw
    return df


def edit_idreams_fatigue(ev, df):
    """
    Real-time fatigue intervention levels.

    Generated features:
        n_fatigue_0 (int): Number of fatigue level 0 events no warning
        n_fatigue_1 (int): Number of fatigue level 1 events visual warning
        n_fatigue_2 (int): Number of fatigue level 2 events visual and
                           auditory warning
        n_fatigue_3 (int): Number of fatigue level 3 events visual and
                           auditory warning

    Args:
        ev (pandas.DataFrame): iDreams_Fatigue_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_fatigue_0 = n_fatigue_1 = n_fatigue_2 = n_fatigue_3 = None
    if ev is not None:
        # number of fatigue level events 0 no warning (Normal Driving)
        n_fatigue_0 = sum(ev['level'] == 0)
        # number of fatigue level events 1 visual warning (Dangerous Driving)
        n_fatigue_1 = sum(ev['level'] == 1)
        # number of fatigue level events 2 visual and auditory warning
        # (Dangerous Driving)
        n_fatigue_2 = sum(ev['level'] == 2)
        # number of fatigue level events 3 visual and auditory warning
        # (Dangerous Driving)
        n_fatigue_3 = sum(ev['level'] == 3)
    # update dataframe
    df['n_fatigue_0'] = n_fatigue_0
    df['n_fatigue_1'] = n_fatigue_1
    df['n_fatigue_2'] = n_fatigue_2
    df['n_fatigue_3'] = n_fatigue_3
    return df


def edit_idreams_headway(ev, df):
    """
    Real-time headway intervention levels.

    Generated features:
        n_headway__1 (int): Number of headway level -1 events no vehicle
                            detected
        n_headway_0 (int): Number of headway level 0 events vehicle detected
        n_headway_1 (int): Number of headway level 1 events vehicle detected
        n_headway_2 (int): Number of headway level 2 events first warning stage
        n_headway_3 (int): Number of headway level 3 events second warning
                           stage

    Args:
        ev (pandas.DataFrame): iDreams_Headway_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_headway__1 = n_headway_0 = n_headway_1 = n_headway_2 = n_headway_3 = None
    if ev is not None:
        # number of headway level -1 no vehicle detected (Normal Driving)
        n_headway__1 = sum(ev['level'] == -1)
        # number of headway level 0 vehicle detected, but headway >= 2.5
        # (Normal Driving)
        n_headway_0 = sum(ev['level'] == 0)
        # number of headway level 1 vehicle detected, headway < 2.5, but above
        # warning threshold (Normal Driving)
        n_headway_1 = sum(ev['level'] == 1)
        # number of headway level 2 first warning stage (Dangerous Driving)
        n_headway_2 = sum(ev['level'] == 2)
        # number of headway level 3 second warning stage (Avoidable Accident)
        n_headway_3 = sum(ev['level'] == 3)
    # update dataframe
    df['n_headway__1'] = n_headway__1
    df['n_headway_0'] = n_headway_0
    df['n_headway_1'] = n_headway_1
    df['n_headway_2'] = n_headway_2
    df['n_headway_3'] = n_headway_3
    return df


def edit_idreams_overtaking(ev, df):
    """
    Real-time overtaking intervention levels.

    Generated features:
        n_overtaking_0 (int): Number of overtaking level 0 events no warning
        n_overtaking_1 (int): Number of overtaking level 1 events visual
                              warning
        n_overtaking_2 (int): Number of overtaking level 2 events visual and
                              auditory warning
        n_overtaking_3 (int): Number of overtaking level 3 events frequent
                              warning

    Args:
        ev (pandas.DataFrame): iDreams_Overtaking_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_overtaking_0 = n_overtaking_1 = n_overtaking_2 = n_overtaking_3 = None
    if ev is not None:
        # number of overtaking level 0 no warning (Normal Driving)
        n_overtaking_0 = sum(ev['level'] == 0)
        # number of overtaking level 1 visual warning (Normal Driving)
        n_overtaking_1 = sum(ev['level'] == 1)
        # number of overtaking level 2 visual and auditory warning
        # (Dangerous Driving)
        n_overtaking_2 = sum(ev['level'] == 2)
        # number of overtaking level 3 frequent warning (Avoidable Accident)
        n_overtaking_3 = sum(ev['level'] == 3)
    # update dataframe
    df['n_overtaking_0'] = n_overtaking_0
    df['n_overtaking_1'] = n_overtaking_1
    df['n_overtaking_2'] = n_overtaking_2
    df['n_overtaking_3'] = n_overtaking_3
    return df


def edit_idreams_speeding(ev, df):
    """
    Real-time speeding intervention levels.

    Generated features:
        n_speeding_0 (int): Number of speeding level 0 events no warning
        n_speeding_1 (int): Number of speeding level 1 events visual indication
        n_speeding_2 (int): Number of speeding level 2 events visual speeding
                            warning
        n_speeding_3 (int): Number of speeding level 3 events visual and
                            auditory warning

    Args:
        ev (pandas.DataFrame): iDreams_Speeding_Map DataFrame
        df (pandas.DataFrame): Dataset DataFrame

    Returns:
        pandas.DataFrame: Dataset updated
    """
    n_speeding_0 = n_speeding_1 = n_speeding_2 = n_speeding_3 = None
    if ev is not None:
        # number of speeding level 0 no warning (Normal Driving)
        n_speeding_0 = sum(ev['level'] == 0)
        # number of speeding level 1 visual indication (Normal Driving)
        n_speeding_1 = sum(ev['level'] == 1)
        # number of speeding level 2 visual speeding warning
        # (Dangerous Driving)
        n_speeding_2 = sum(ev['level'] == 2)
        # number of speeding level 3 visual and auditory warning
        # (Avoidable Accident)
        n_speeding_3 = sum(ev['level'] == 3)
    # update dataframe
    df['n_speeding_0'] = n_speeding_0
    df['n_speeding_1'] = n_speeding_1
    df['n_speeding_2'] = n_speeding_2
    df['n_speeding_3'] = n_speeding_3
    return df


if __name__ == "__main__":

    # folder = '../trips/2021_04_21T05_42_57__25Kf7/'
    # info = read_json_file(folder + 'info.json')

    pull_trips_and_store_df(
        '../datasets/constructed', 'trips_test_drivers_2022-07-21_2022-08-20', '2022-07-21', None
    )

    # df1_name = '../datasets/constructed/trips_2021-04-1_2021-07-1'
    # df2_name = '../datasets/constructed/trips_2021-07-2_2021-10-1'
    # df1 = read_csv_file(df1_name)
    # df2 = read_csv_file(df2_name)
    # trips = join_datasets(df1, df2)

    # df2_name = '../datasets/constructed/trips_2021-10-2_2022-01-1'
    # df2 = read_csv_file(df2_name)
    # trips = join_datasets(trips, df2)

    # df2_name = '../datasets/constructed/trips_2022-01-2_2022-04-1'
    # df2 = read_csv_file(df2_name)
    # trips = join_datasets(trips, df2)

    # df2_name = '../datasets/constructed/trips_2022-04-2_2022-05-13'
    # df2 = read_csv_file(df2_name)
    # trips = join_datasets(trips, df2)

    # print(trips.shape)

    # store_csv('../datasets/constructed', 'trips_until_2022-05-13', trips)

    # trips_v2   -> Uma feature estava mal calculada
    # trips_v2.1 -> Timedeltas converted to seconds
    # trips_v2.2 -> light_mode converted to int (day=1, dusk=2, night=3)

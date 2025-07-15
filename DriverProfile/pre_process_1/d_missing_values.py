"""
preprocess.missing_values
-------

This module provides diferent aproaches to deal with missing values:
    1 - Remove rows and columns (
        with all values missing or
        distance=0 or
        duration=0
    )
    2 - Fill with a value
    3 - Fill with mean
    4 - Fill with median
    5 - Fill with mode
    6 - Fill with new category (for categorical)
    7 - Fill with Last observation carried forward (LOCF)
    8 - Predict the value with ml algorithms ?

Strategy taken (when it is not impossible to calculate a value):
    1 - Check the percentage of missing values in all columns
    2 - If is <= 10%: Fill with mean or median or mode
    3 - If is >= 50%: Drop column
"""

# packages
import pandas as pd
# local
from .b_construct_dataset import read_csv_file, store_csv
from .d_trip_light_mode import get_light_mode


def fill_hands_on_missing_values(df, column_names):
    if 'n_lod_0' in column_names:
        has, per = has_too_many_nulls(df, 'n_lod_0')
        if not has:
            df['n_lod_0'] = df['n_lod_0'].fillna(df['n_lod_0'].median())
        elif per >= 0.5:
            del df['n_lod_0']
    if 'n_lod_1' in column_names:
        has, per = has_too_many_nulls(df, 'n_lod_1')
        if not has:
            df['n_lod_1'] = df['n_lod_1'].fillna(df['n_lod_1'].median())
        elif per >= 0.5:
            del df['n_lod_1']
    if 'n_lod_2' in column_names:
        has, per = has_too_many_nulls(df, 'n_lod_2')
        if not has:
            df['n_lod_2'] = df['n_lod_2'].fillna(df['n_lod_2'].median())
        elif per >= 0.5:
            del df['n_lod_2']
    if 'n_lod_3' in column_names:
        has, per = has_too_many_nulls(df, 'n_lod_3')
        if not has:
            df['n_lod_3'] = df['n_lod_3'].fillna(df['n_lod_3'].median())
        elif per >= 0.5:
            del df['n_lod_3']
    return df


def fill_drowsiness_missing_values(df, column_names):
    if 'n_drowsiness_0' in column_names:
        has, per = has_too_many_nulls(df, 'n_drowsiness_0')
        if not has:
            df['n_drowsiness_0'] = df['n_drowsiness_0'] \
                .fillna(df['n_drowsiness_0'].median())
        elif per >= 0.5:
            del df['n_drowsiness_0']
    if 'n_drowsiness_1' in column_names:
        has, per = has_too_many_nulls(df, 'n_drowsiness_1')
        if not has:
            df['n_drowsiness_1'] = df['n_drowsiness_1'] \
                .fillna(df['n_drowsiness_1'].median())
        elif per >= 0.5:
            del df['n_drowsiness_1']
    if 'n_drowsiness_2' in column_names:
        has, per = has_too_many_nulls(df, 'n_drowsiness_2')
        if not has:
            df['n_drowsiness_2'] = df['n_drowsiness_2'] \
                .fillna(df['n_drowsiness_2'].median())
        elif per >= 0.5:
            del df['n_drowsiness_2']
    if 'n_drowsiness_3' in column_names:
        has, per = has_too_many_nulls(df, 'n_drowsiness_3')
        if not has:
            df['n_drowsiness_3'] = df['n_drowsiness_3'] \
                .fillna(df['n_drowsiness_3'].median())
        elif per >= 0.5:
            del df['n_drowsiness_3']
    return df


def fill_driving_events_missing_values(df, column_names):
    if 'n_ha' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_ha')
        if not has:
            df['n_ha'] = df['n_ha'].fillna(df['n_ha'].median())
        elif per >= 0.5:
            del df['n_ha']
    if 'n_ha_l' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_ha_l')
        if not has:
            df['n_ha_l'] = df['n_ha_l'].fillna(df['n_ha_l'].median())
        elif per >= 0.5:
            del df['n_ha_l']
    if 'n_ha_m' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_ha_m')
        if not has:
            df['n_ha_m'] = df['n_ha_m'].fillna(df['n_ha_m'].median())
        elif per >= 0.5:
            del df['n_ha_m']
    if 'n_ha_h' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_ha_h')
        if not has:
            df['n_ha_h'] = df['n_ha_h'].fillna(df['n_ha_h'].median())
        elif per >= 0.5:
            del df['n_ha_h']
    if 'n_hb' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_hb')
        if not has:
            df['n_hb'] = df['n_hb'].fillna(df['n_hb'].median())
        elif per >= 0.5:
            del df['n_hb']
    if 'n_hb_l' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_hb_l')
        if not has:
            df['n_hb_l'] = df['n_hb_l'].fillna(df['n_hb_l'].median())
        elif per >= 0.5:
            del df['n_hb_l']
    if 'n_hb_m' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_hb_m')
        if not has:
            df['n_hb_m'] = df['n_hb_m'].fillna(df['n_hb_m'].median())
        elif per >= 0.5:
            del df['n_hb_m']
    if 'n_hb_h' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_hb_h')
        if not has:
            df['n_hb_h'] = df['n_hb_h'].fillna(df['n_hb_h'].median())
        elif per >= 0.5:
            del df['n_hb_h']
    if 'n_hc' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_hc')
        if not has:
            df['n_hc'] = df['n_hc'].fillna(df['n_hc'].median())
        elif per >= 0.5:
            del df['n_hc']
    if 'n_hc_l' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_hc_l')
        if not has:
            df['n_hc_l'] = df['n_hc_l'].fillna(df['n_hc_l'].median())
        elif per >= 0.5:
            del df['n_hc_l']
    if 'n_hc_m' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_hc_m')
        if not has:
            df['n_hc_m'] = df['n_hc_m'].fillna(df['n_hc_m'].median())
        elif per >= 0.5:
            del df['n_hc_m']
    if 'n_hc_h' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_hc_h')
        if not has:
            df['n_hc_h'] = df['n_hc_h'].fillna(df['n_hc_h'].median())
        elif per >= 0.5:
            del df['n_hc_h']
    return df


def fill_distraction_missing_values(df, column_names):
    if 'distraction_time' in column_names:
        has, per = has_too_many_nulls(df, 'distraction_time')
        if not has:
            df['distraction_time'] = df['distraction_time'] \
                .fillna(df['distraction_time'].median())
        elif per >= 0.5:
            del df['distraction_time']
    if 'n_distractions' in column_names:
        has, per = has_too_many_nulls(df, 'n_distractions')
        if not has:
            df['n_distractions'] = df['n_distractions'] \
                .fillna(df['n_distractions'].median())
        elif per >= 0.5:
            del df['n_distractions']
    return df


def fill_ignition_missing_values(df, column_names):
    if 'n_ignition_on' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'n_ignition_on')
        if not has:
            df['n_ignition_on'] = df['n_ignition_on'] \
                .fillna(df['n_ignition_on'].median())
        elif per >= 0.5:
            del df['n_ignition_on']
    if 'n_ignition_off' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'n_ignition_off')
        if not has:
            df['n_ignition_off'] = df['n_ignition_off'] \
                .fillna(df['n_ignition_off'].median())
        elif per >= 0.5:
            del df['n_ignition_off']
    return df


def fill_me_aws_missing_values(df, column_names):
    if 'fcw_time' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'fcw_time')
        if not has:
            df['fcw_time'] = df['fcw_time'].fillna(df['fcw_time'].median())
        elif per >= 0.5:
            del df['fcw_time']
    if 'hmw_time' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'hmw_time')
        if not has:
            df['hmw_time'] = df['hmw_time'].fillna(df['hmw_time'].median())
        elif per >= 0.5:
            del df['hmw_time']
    if 'ldw_time' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'ldw_time')
        if not has:
            df['ldw_time'] = df['ldw_time'].fillna(df['ldw_time'].median())
        elif per >= 0.5:
            del df['ldw_time']
    if 'pcw_time' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'pcw_time')
        if not has:
            df['pcw_time'] = df['pcw_time'].fillna(df['pcw_time'].median())
        elif per >= 0.5:
            del df['pcw_time']
    if 'n_pedestrian_dz' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_pedestrian_dz')
        if not has:
            df['n_pedestrian_dz'] = df['n_pedestrian_dz'] \
                .fillna(df['n_pedestrian_dz'].median())
        elif per >= 0.5:
            del df['n_pedestrian_dz']
    if 'light_mode' in column_names:
        dataset = df[df['light_mode'].isnull()][[
            'trip_start', 'trip_end', 'light_mode'
        ]]
        dataset['trip_start'] = pd.to_datetime(dataset['trip_start'])
        dataset['trip_end'] = pd.to_datetime(dataset['trip_end'])
        #for i, row in dataset.iterrows():
        #    dataset.at[i, 'light_mode'] = get_light_mode(row)
        df['light_mode'] = df['light_mode'].fillna(dataset['light_mode'])
    if 'n_tsr_level' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level')
        if not has:
            df['n_tsr_level'] = df['n_tsr_level'] \
                .fillna(df['n_tsr_level'].median())
        elif per >= 0.5:
            del df['n_tsr_level']
    if 'n_tsr_level_0' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_0')
        if not has:
            df['n_tsr_level_0'] = df['n_tsr_level_0'] \
                .fillna(df['n_tsr_level_0'].median())
        elif per >= 0.5:
            del df['n_tsr_level_0']
    if 'n_tsr_level_1' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_1')
        if not has:
            df['n_tsr_level_1'] = df['n_tsr_level_1'] \
                .fillna(df['n_tsr_level_1'].median())
        elif per >= 0.5:
            del df['n_tsr_level_1']
    if 'n_tsr_level_2' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_2')
        if not has:
            df['n_tsr_level_2'] = df['n_tsr_level_2'] \
                .fillna(df['n_tsr_level_2'].median())
        elif per >= 0.5:
            del df['n_tsr_level_2']
    if 'n_tsr_level_3' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_3')
        if not has:
            df['n_tsr_level_3'] = df['n_tsr_level_3'] \
                .fillna(df['n_tsr_level_3'].median())
        elif per >= 0.5:
            del df['n_tsr_level_3']
    if 'n_tsr_level_4' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_4')
        if not has:
            df['n_tsr_level_4'] = df['n_tsr_level_4'] \
                .fillna(df['n_tsr_level_4'].median())
        elif per >= 0.5:
            del df['n_tsr_level_4']
    if 'n_tsr_level_5' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_5')
        if not has:
            df['n_tsr_level_5'] = df['n_tsr_level_5'] \
                .fillna(df['n_tsr_level_5'].median())
        elif per >= 0.5:
            del df['n_tsr_level_5']
    if 'n_tsr_level_6' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_6')
        if not has:
            df['n_tsr_level_6'] = df['n_tsr_level_6'] \
                .fillna(df['n_tsr_level_6'].median())
        elif per >= 0.5:
            del df['n_tsr_level_6']
    if 'n_tsr_level_7' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_tsr_level_7')
        if not has:
            df['n_tsr_level_7'] = df['n_tsr_level_7'] \
                .fillna(df['n_tsr_level_7'].median())
        elif per >= 0.5:
            del df['n_tsr_level_7']
    if 'zero_speed_time' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'zero_speed_time')
        if not has:
            df['zero_speed_time'] = df['zero_speed_time'] \
                .fillna(df['zero_speed_time'].median())
        elif per >= 0.5:
            del df['zero_speed_time']
    if 'n_zero_speed' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_zero_speed')
        if not has:
            df['n_zero_speed'] = df['n_zero_speed'] \
                .fillna(df['n_zero_speed'].median())
        elif per >= 0.5:
            del df['n_zero_speed']
    if 'lat' in df.columns:
        del df['lat']
    if 'lon' in df.columns:
        del df['lon']
    return df


def fill_me_car_missing_values(df, column_names):
    if 'n_high_beam' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'n_high_beam')
        if not has:
            df['n_high_beam'] = df['n_high_beam'] \
                .fillna(df['n_high_beam'].median())
        elif per >= 0.5:
            del df['n_high_beam']
    if 'n_low_beam' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'n_low_beam')
        if not has:
            df['n_low_beam'] = df['n_low_beam'] \
                .fillna(df['n_low_beam'].median())
        elif per >= 0.5:
            del df['n_low_beam']
    if 'n_wipers' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'n_wipers')
        if not has:
            df['n_wipers'] = df['n_wipers'].fillna(df['n_wipers'].median())
        elif per >= 0.5:
            del df['n_wipers']
    if 'n_signal_right' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_signal_right')
        if not has:
            df['n_signal_right'] = df['n_signal_right'] \
                .fillna(df['n_signal_right'].median())
        elif per >= 0.5:
            del df['n_signal_right']
    if 'n_signal_left' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_signal_left')
        if not has:
            df['n_signal_left'] = df['n_signal_left'] \
                .fillna(df['n_signal_left'].median())
        elif per >= 0.5:
            del df['n_signal_left']
    if 'n_brakes' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_brakes')
        if not has:
            df['n_brakes'] = df['n_brakes'].fillna(df['n_brakes'].median())
        elif per >= 0.5:
            del df['n_brakes']
    if 'speed' in column_names:
        # mean
        has, per = has_too_many_nulls(df, 'speed')
        if not has:
            df['speed'] = df['speed'].fillna(df['speed'].mean())
        elif per >= 0.5:
            del df['speed']
    if 'over_speed_limit' in column_names:
        has, per = has_too_many_nulls(df, 'over_speed_limit')
        if not has:
            df['over_speed_limit'] = df['over_speed_limit'] \
                .fillna(df['over_speed_limit'].median())
        elif per >= 0.5:
            del df['over_speed_limit']
    return df


def fill_me_fcw_missing_values(df, column_names):
    if 'n_fcw' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_fcw')
        if not has:
            df['n_fcw'] = df['n_fcw'].fillna(df['n_fcw'].median())
        elif per >= 0.5:
            del df['n_fcw']
    return df


def fill_me_hmw_missing_values(df, column_names):
    if 'n_hmw' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_hmw')
        if not has:
            df['n_hmw'] = df['n_hmw'].fillna(df['n_hmw'].median())
        elif per >= 0.5:
            del df['n_hmw']
    return df


def fill_me_ldw_missing_values(df, column_names):
    if 'n_ldw' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_ldw')
        if not has:
            df['n_ldw'] = df['n_ldw'].fillna(df['n_ldw'].median())
        elif per >= 0.5:
            del df['n_ldw']
    if 'n_ldw_left' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_ldw_left')
        if not has:
            df['n_ldw_left'] = df['n_ldw_left'] \
                .fillna(df['n_ldw_left'].median())
        elif per >= 0.5:
            del df['n_ldw_left']
    if 'n_ldw_right' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_ldw_right')
        if not has:
            df['n_ldw_right'] = df['n_ldw_right'] \
                .fillna(df['n_ldw_right'].median())
        elif per >= 0.5:
            del df['n_ldw_right']
    return df


def fill_me_pcw_missing_values(df, column_names):
    if 'n_pcw' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_pcw')
        if not has:
            df['n_pcw'] = df['n_pcw'].fillna(df['n_pcw'].median())
        elif per >= 0.5:
            del df['n_pcw']
    return df


def fill_idreams_fatigue_missing_values(df, column_names):
    if 'n_fatigue_0' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_fatigue_0')
        if not has:
            df['n_fatigue_0'] = df['n_fatigue_0'] \
                .fillna(df['n_fatigue_0'].median())
        elif per >= 0.5:
            del df['n_fatigue_0']
    if 'n_fatigue_1' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_fatigue_1')
        if not has:
            df['n_fatigue_1'] = df['n_fatigue_1'] \
                .fillna(df['n_fatigue_1'].median())
        elif per >= 0.5:
            del df['n_fatigue_1']
    if 'n_fatigue_2' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_fatigue_2')
        if not has:
            df['n_fatigue_2'] = df['n_fatigue_2'] \
                .fillna(df['n_fatigue_2'].median())
        elif per >= 0.5:
            del df['n_fatigue_2']
    if 'n_fatigue_3' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_fatigue_3')
        if not has:
            df['n_fatigue_3'] = df['n_fatigue_3'] \
                .fillna(df['n_fatigue_3'].median())
        elif per >= 0.5:
            del df['n_fatigue_3']
    return df


def fill_idreams_headway_missing_values(df, column_names):
    if 'n_headway__1' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_headway__1')
        if not has:
            df['n_headway__1'] = df['n_headway__1'] \
                .fillna(df['n_headway__1'].median())
        elif per >= 0.5:
            del df['n_headway__1']
    if 'n_headway_0' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_headway_0')
        if not has:
            df['n_headway_0'] = df['n_headway_0'] \
                .fillna(df['n_headway_0'].median())
        elif per >= 0.5:
            del df['n_headway_0']
    if 'n_headway_1' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_headway_1')
        if not has:
            df['n_headway_1'] = df['n_headway_1'] \
                .fillna(df['n_headway_1'].median())
        elif per >= 0.5:
            del df['n_headway_1']
    if 'n_headway_2' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_headway_2')
        if not has:
            df['n_headway_2'] = df['n_headway_2'] \
                .fillna(df['n_headway_2'].median())
        elif per >= 0.5:
            del df['n_headway_2']
    if 'n_headway_3' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_headway_3')
        if not has:
            df['n_headway_3'] = df['n_headway_3'] \
                .fillna(df['n_headway_3'].median())
        elif per >= 0.5:
            del df['n_headway_3']
    return df


def fill_idreams_overtaking_missing_values(df, column_names):
    if 'n_overtaking_0' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_overtaking_0')
        if not has:
            df['n_overtaking_0'] = df['n_overtaking_0'] \
                .fillna(df['n_overtaking_0'].median())
        elif per >= 0.5:
            del df['n_overtaking_0']
    if 'n_overtaking_1' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_overtaking_1')
        if not has:
            df['n_overtaking_1'] = df['n_overtaking_1'] \
                .fillna(df['n_overtaking_1'].median())
        elif per >= 0.5:
            del df['n_overtaking_1']
    if 'n_overtaking_2' in column_names:
        # median or mean (only 1 outlier)
        has, per = has_too_many_nulls(df, 'n_overtaking_2')
        if not has:
            df['n_overtaking_2'] = df['n_overtaking_2'] \
                .fillna(df['n_overtaking_2'].median())
        elif per >= 0.5:
            del df['n_overtaking_2']
    if 'n_overtaking_3' in column_names:
        # median or mean
        has, per = has_too_many_nulls(df, 'n_overtaking_3')
        if not has:
            df['n_overtaking_3'] = df['n_overtaking_3'] \
                .fillna(df['n_overtaking_3'].median())
        elif per >= 0.5:
            del df['n_overtaking_3']
    return df


def fill_idreams_speeding_missing_values(df, column_names):
    if 'n_speeding_0' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_speeding_0')
        if not has:
            df['n_speeding_0'] = df['n_speeding_0'] \
                .fillna(df['n_speeding_0'].median())
        elif per >= 0.5:
            del df['n_speeding_0']
    if 'n_speeding_1' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_speeding_1')
        if not has:
            df['n_speeding_1'] = df['n_speeding_1'] \
                .fillna(df['n_speeding_1'].median())
        elif per >= 0.5:
            del df['n_speeding_1']
    if 'n_speeding_2' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_speeding_2')
        if not has:
            df['n_speeding_2'] = df['n_speeding_2'] \
                .fillna(df['n_speeding_2'].median())
        elif per >= 0.5:
            del df['n_speeding_2']
    if 'n_speeding_3' in column_names:
        # median
        has, per = has_too_many_nulls(df, 'n_speeding_3')
        if not has:
            df['n_speeding_3'] = df['n_speeding_3'] \
                .fillna(df['n_speeding_3'].median())
        elif per >= 0.5:
            del df['n_speeding_3']
    return df


def delete_missing_values(df):
    df = df.dropna(axis=1, how='all')  # columns with all NaN
    df = df.dropna(how='all')  # rows with all NaN
    # rows with all NaN (dont count start, end, distance and duration)
    df_mid = df.iloc[:, 4:].dropna(how='all')
    df = df.iloc[:, :4].join(df_mid)
    # remove distance and duration <= 0
    df = df[df.distance > 0]
    df = df[df.duration > 0]
    # remove trips where duration is less than 1 minute
    df = df[df.duration >= 60]
    # remove trips where distance is less than 1.5 kms
    df = df[df.distance >= 1]
    return df


def has_too_many_nulls(df, column_name, percent=0.1):
    n_values = df[column_name].isnull().sum()
    return (n_values / len(df[column_name])) > percent, n_values / len(df[column_name])


def check_columns_with_nulls(df):
    """
    Check what features have missing values

    Args:
        df (pandas.DataFrame): Dataset

    Returns:
        list: Column names that have missing values
    """
    # check columns with null values
    columns_nan = df.columns[df.isnull().any()].tolist()
    return columns_nan


def fill_missing_values(trips, columns_nan):

    # fill missing values
    trips = fill_hands_on_missing_values(trips, columns_nan)
    trips = fill_drowsiness_missing_values(trips, columns_nan)
    trips = fill_driving_events_missing_values(trips, columns_nan)
    trips = fill_distraction_missing_values(trips, columns_nan)
    trips = fill_ignition_missing_values(trips, columns_nan)
    trips = fill_me_aws_missing_values(trips, columns_nan)
    trips = fill_me_car_missing_values(trips, columns_nan)
    trips = fill_me_fcw_missing_values(trips, columns_nan)
    trips = fill_me_hmw_missing_values(trips, columns_nan)
    trips = fill_me_ldw_missing_values(trips, columns_nan)
    trips = fill_me_pcw_missing_values(trips, columns_nan)
    trips = fill_idreams_fatigue_missing_values(trips, columns_nan)
    trips = fill_idreams_headway_missing_values(trips, columns_nan)
    trips = fill_idreams_overtaking_missing_values(trips, columns_nan)
    trips = fill_idreams_speeding_missing_values(trips, columns_nan)
    return trips



if __name__ == "__main__":

    # read dataset
    #trips = read_csv_file('../datasets/constructed/trips_test_2022-05-14_2022-07-20')
    trips = pd.read_csv('../datasets/constructed/trips_until_2022-05-13.csv')
    #df1 = read_csv_file('../datasets/missing_values/trips_mv')
    #df2 = read_csv_file('../datasets/missing_values/trips_mv_test')
    #trips = pd.concat([df1, df2], ignore_index=True)

    print(trips.shape)


    # remove columns/rows that have all values NaN
    trips = delete_missing_values(trips)

    # check columns with null values
    columns_nan = check_columns_with_nulls(trips)
    
    print("Null conumns original: ", len(columns_nan))

    # fill missing values
    trips = fill_hands_on_missing_values(trips, columns_nan)
    trips = fill_drowsiness_missing_values(trips, columns_nan)
    trips = fill_driving_events_missing_values(trips, columns_nan)
    trips = fill_distraction_missing_values(trips, columns_nan)
    trips = fill_ignition_missing_values(trips, columns_nan)
    trips = fill_me_aws_missing_values(trips, columns_nan)
    trips = fill_me_car_missing_values(trips, columns_nan)
    trips = fill_me_fcw_missing_values(trips, columns_nan)
    trips = fill_me_hmw_missing_values(trips, columns_nan)
    trips = fill_me_ldw_missing_values(trips, columns_nan)
    trips = fill_me_pcw_missing_values(trips, columns_nan)
    trips = fill_idreams_fatigue_missing_values(trips, columns_nan)
    trips = fill_idreams_headway_missing_values(trips, columns_nan)
    trips = fill_idreams_overtaking_missing_values(trips, columns_nan)
    trips = fill_idreams_speeding_missing_values(trips, columns_nan)

    # check columns with null values after inputing null values
    columns_nan = check_columns_with_nulls(trips)

    #print(trips)
    #print(trips.columns)

    if len(columns_nan) == 1:
        # store dataset
        print(columns_nan)
        store_csv('../datasets/missing_values', 'trips_mv_all', trips)
    print("Null conumns: ", columns_nan)
import pandas as pd
import datetime
from astral import LocationInfo
from astral.sun import sun, night, midnight


def get_light_mode(trip):
    """
    Calculates whether the trip took place mostly during the day, dusk or night

    Args:
        trip (pandas.Series): Trip instance

    Returns:
        str: Trip lightning condition
    """

    start = pd.to_datetime(trip['trip_start'])
    end = pd.to_datetime(trip['trip_end'])
    trip_interval = pd.Interval(pd.Timestamp(start), pd.Timestamp(end))

    city = LocationInfo("", "", "", trip['lat'], trip['lon'])

    # trip in the same day
    if start.date() == end.date():
        s = sun(city.observer, date=start)
        night_start, night_end = night(city.observer, date=start)

    # trip in 2 different days       
    else:
        mid_night = midnight(city.observer, date=start)
        diff_first = ((mid_night - start) + datetime.timedelta(days=2)).total_seconds()
        diff_second = (end - mid_night).total_seconds()
        # trip takes more time in second day
        if diff_first < diff_second:
            s = sun(city.observer, date=end)
            night_start, night_end = night(city.observer, date=end)
        else:
            s = sun(city.observer, date=start)
            night_start, night_end = night(city.observer, date=start)

    day_start = s["dawn"] + pd.Timedelta('00:00:01')
    day_end = s["sunset"]
    dusk_start = s["sunset"] + pd.Timedelta('00:00:01')
    dusk_end = s["dusk"] - pd.Timedelta('00:00:01')

    day_interval = pd.Interval(pd.Timestamp(day_start), pd.Timestamp(day_end))
    dusk_interval = pd.Interval(pd.Timestamp(dusk_start), pd.Timestamp(dusk_end))
    night_interval = pd.Interval(pd.Timestamp(night_start), pd.Timestamp(night_end))
    day_delta = dusk_delta = night_delta = None

    if trip_interval.overlaps(day_interval):
        latest_start = max(day_start, start)
        earliest_end = min(day_end, end)
        day_delta = (earliest_end - latest_start).total_seconds()

    if trip_interval.overlaps(dusk_interval):
        latest_start = max(dusk_start, start)
        earliest_end = min(dusk_end, end)
        dusk_delta = (earliest_end - latest_start).total_seconds()

    if trip_interval.overlaps(night_interval):
        latest_start = max(night_start, start)
        earliest_end = min(night_end, end)
        night_delta = (earliest_end - latest_start).total_seconds()

    # get the time of day where the trip took the most time
    time_list = [day_delta, dusk_delta, night_delta]
    max_time = max(list(filter(None, time_list)))

    if max_time == day_delta:
        return 'day'
    if max_time == dusk_delta:
        return 'dusk'
    if max_time == night_delta:
        return 'night'

        
if __name__ == "__main__":

    d = {
        'trip_start': [
            pd.to_datetime('2022-05-16T19:16:25+00:00'),
            pd.to_datetime('2022-05-18T19:38:37+00:00'),
        ],
        'trip_end': [
            pd.to_datetime('2022-05-16T20:03:12+00:00'),
            pd.to_datetime('2022-05-18T19:55:22+00:00'),
        ],
        'light_mode': [None, None],
        'lat': [50.951448, 51.016454],
        'lon': [5.309245, 5.288602],
    }

    # 1312
    # 1318

    df = pd.DataFrame(data=d)
    print('data_frame', df, "\n")

    for index, row in df.iterrows():
        light = get_light_mode(row)
        print(light)
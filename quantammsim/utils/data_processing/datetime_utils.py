import pandas as pd
from datetime import datetime, timezone


def unixtimestamp_to_datetime(unixtimestamp, scaling=1000):
    return datetime.utcfromtimestamp(int(unixtimestamp / scaling)).strftime(
        "%Y-%m-%d-%H-%M"
    )


def unixtimestamp_to_precise_datetime(unixtimestamp, scaling=1000):
    return datetime.utcfromtimestamp(int(unixtimestamp / scaling)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def unixtimestamp_to_minute_datetime(unixtimestamp, scaling=1000):
    return datetime.utcfromtimestamp(int(unixtimestamp / scaling)).strftime(
        "%Y-%m-%d %H:%M"
    )


def unixtimestamp_to_midnight_datetime(unixtimestamp, scaling=1000):
    return (
        datetime.utcfromtimestamp(int(unixtimestamp / scaling))
        .replace(hour=0, minute=0, second=0)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def datetime_to_unixtimestamp(date_time, str_format="%Y-%m-%d %H:%M"):
    return datetime.timestamp(
        datetime.strptime(date_time, str_format).replace(tzinfo=timezone.utc)
    )


def pddatetime_to_unixtimestamp(date_time):
    return date_time.values.astype(np.int64) // 10**9

'''
DataStore is a class that stores data fetched from a data provider.

It can store it in flat files, in a database, or in memory; whatever the developer wants.

API change:
- get(identifier, dt) -> dict | None  (single point-in-time)
- get_all(identifier, start_date, end_date) -> OrderedDict[datetime, dict] | None (range)
'''

from datetime import datetime
from typing import Union, OrderedDict
from collections import OrderedDict as OrderedDictType
import abc
import os
import pandas as pd
from pathlib import Path
import numpy as np

'''

Data format:

identifier: {
    datetime_str: {
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    },
    datetime_str: {
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    },
    ...
}

Could also be fundamental data, etc. like:

identifier: {
    datetime_str: {
        gdp: float,
        inflation: float,
        unemployment: float,
        interest_rate: float,
        etc.
    },
    datetime_str: {
        gdp: float,
        inflation: float,
        unemployment: float,
        interest_rate: float,
        etc.
    },
    ...
}

'''

class DataStore(abc.ABC):
    def __init__(self, *args, **kwargs):
        ...

    '''
    Adds data to the store. Assumes that data is immutable, and that if the data already exists for the given times, 
    it is skipped.
    '''
    def add(self, identifier: str, data: dict):
        ...

    '''
    Get a single point-in-time record for the identifier. Returns None if not present.
    '''
    def get(self, identifier: str, dt: datetime) -> Union[dict, None]:
        ...

    '''
    Gets data from the store in chronological order (earliest to latest) for a range.
    If the data is not found, it returns None.
    '''
    def get_all(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None) -> Union[OrderedDictType[datetime, dict], None]:
        ...

    '''
    Gets the latest data point for the identifier.
    If the data is not found, it returns None.
    '''
    def get_latest(self, identifier: str) -> Union[dict, None]:
        ...

    '''
    Gets the latest date for a given identifier from the store.
    If the data is not found, it returns None.
    '''
    def latest(self, identifier: str) -> Union[datetime, None]:
        ...

    '''
    Gets the earliest date for a given identifier from the store.
    If the data is not found, it returns None.
    '''
    def earliest(self, identifier: str) -> Union[datetime, None]:
        ...

    '''
    Checks if data exists for a given identifier, start_date, and end_date.

    If start_date is None, it assumes the earliest date for that piece of data.
    If end_date is not None, it checks if the data exists
    for the given start_date until the end of the data.
    '''
    def exists(self, identifier: str, start_date: Union[datetime, None] = None, end_date: Union[datetime, None] = None) -> bool:
        ...
    
    '''
    Gets all identifiers from the store.
    '''
    def identifiers(self):
        ...




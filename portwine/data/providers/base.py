import abc
from datetime import datetime
from typing import Union


class DataProvider(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    def get_data(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        # gets for a given identifier, start_date, and end_date
        # data can be ANY format, OHLCV, fundamental data, etc.
        # this is just the interface for the data provider
        ...

    async def get_data_async(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        # gets for a given identifier, start_date, and end_date
        # data can be ANY format, OHLCV, fundamental data, etc.
        # this is just the interface for the data provider
        ...

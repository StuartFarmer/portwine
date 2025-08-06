import abc
from datetime import datetime
from typing import Union
import httpx
import asyncio

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

'''
EODHD Historical Data Provider

This provider is used to get historical data from EODHD.

https://eodhd.com/

import requests

url = f'https://eodhd.com/api/eod/MCD.US?api_token=67740bda7e4247.39007920&fmt=json'
data = requests.get(url).json()

print(data)
'''
class EODHDHistoricalDataProvider(DataProvider):
    def __init__(self, api_key: str, exchange_code: str):
        self.api_key = api_key
        self.exchange_code = exchange_code

    def _get_url(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        url = f'https://eodhd.com/api/eod/{identifier}.{self.exchange_code}?api_token={self.api_key}&fmt=json'
        if end_date is not None:
            url += f'&to={end_date.strftime("%Y-%m-%d")}'
        if start_date is not None:
            url += f'&from={start_date.strftime("%Y-%m-%d")}'
        return url

    def get_data(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        url = self._get_url(identifier, start_date, end_date)
        data = httpx.get(url).json()
        
        return data
    
    async def get_data_async(self, identifier: str, start_date: datetime, end_date: Union[datetime, None] = None):
        url = self._get_url(identifier, start_date, end_date)
        
        async with httpx.AsyncClient() as client:
            data = await client.get(url)
            data = data.json()

        return data
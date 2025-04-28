import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import requests
from portwine.loaders.polygon import PolygonMarketDataLoader
import pytz
from freezegun import freeze_time


@pytest.fixture
def loader():
    """Create a PolygonMarketDataLoader instance for testing."""
    return PolygonMarketDataLoader(api_key="test_key", data_dir="test_data")


@pytest.fixture
def mock_response():
    """Create a mock response with sample data."""
    return {
        "results": [
            {
                "t": 1609459200000,  # 2021-01-01
                "o": 100.0,
                "h": 105.0,
                "l": 95.0,
                "c": 102.0,
                "v": 1000
            },
            {
                "t": 1609545600000,  # 2021-01-02
                "o": 102.0,
                "h": 107.0,
                "l": 97.0,
                "c": 104.0,
                "v": 1100
            }
        ]
    }


def test_api_get_url_is_none(loader):
    """Test that _api_get raises an exception when URL is None."""
    with pytest.raises(ValueError):
        loader._api_get(None)


def test_api_get_valid_response(loader, mock_response):
    """Test that _api_get returns valid JSON response."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        response = loader._api_get("test_url")
        assert response == mock_response


def test_api_get_response_raises_exception(loader):
    """Test that _api_get bubbles up HTTP errors."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        
        with pytest.raises(requests.HTTPError):
            loader._api_get("test_url")


def test_api_get_RequestException(loader):
    """Test that _api_get raises exception on RequestException."""
    with patch('requests.Session.get') as mock_get:
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(requests.RequestException):
            loader._api_get("test_url")


def test_fetch_historical_data_malformed_to_date(loader):
    """Test that fetch_historical_data raises error for malformed to_date."""
    with pytest.raises(ValueError):
        loader.fetch_historical_data("AAPL", to_date="invalid_date")


def test_fetch_historical_data_malformed_from_date(loader):
    """Test that fetch_historical_data raises error for malformed from_date."""
    with pytest.raises(ValueError):
        loader.fetch_historical_data("AAPL", from_date="invalid_date")


def test_fetch_historical_data_to_is_after_from_date(loader):
    """Test that fetch_historical_data raises error when to_date is after from_date."""
    with pytest.raises(ValueError):
        loader.fetch_historical_data(
            "AAPL",
            from_date="2021-01-02",
            to_date="2021-01-01"
        )


def test_fetch_historical_data_works(loader, mock_response):
    """Test that fetch_historical_data returns data for the entire period."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        df = loader.fetch_historical_data("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']


def test_fetch_historical_data_adds_to_cache(loader, mock_response):
    """Test that fetch_historical_data adds data to in-memory cache."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        loader.fetch_historical_data("AAPL")
        assert "AAPL" in loader._data_cache
        assert isinstance(loader._data_cache["AAPL"], pd.DataFrame)


def test_fetch_historical_data_adds_saves_to_disk(loader, mock_response):
    """Test that fetch_historical_data saves data to disk."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        loader.fetch_historical_data("AAPL")
        data_path = os.path.join("test_data", "AAPL.parquet")
        assert os.path.exists(data_path)
        
        # Clean up
        os.remove(data_path)
        os.rmdir("test_data")


def test_fetch_historical_data_invalid_ticker(loader):
    """Test that fetch_historical_data handles invalid ticker."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = {"error": "Invalid ticker"}
        mock_get.return_value.raise_for_status = MagicMock()
        
        result = loader.fetch_historical_data("INVALID")
        assert result is None


def test_fetch_historical_data_logs_exceptions(loader, caplog):
    """Test that fetch_historical_data logs exceptions and returns None."""
    with patch('requests.Session.get') as mock_get:
        mock_get.side_effect = Exception("Test error")
        
        result = loader.fetch_historical_data("AAPL")
        assert result is None
        assert "Error fetching historical data for AAPL" in caplog.text


def test_init_if_no_api_key_warn(caplog):
    """Test that init warns if no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            PolygonMarketDataLoader()
        assert "Polygon API key not provided" in caplog.text


def test_get_data_path_valid_ticker(loader):
    """Test that _get_data_path returns expected path for valid ticker."""
    path = loader._get_data_path("AAPL")
    assert path == os.path.join("test_data", "AAPL.parquet")


def test_get_data_path_invalid_ticker(loader):
    """Test that _get_data_path raises error for invalid ticker characters."""
    with pytest.raises(ValueError):
        loader._get_data_path("AAPL/INVALID")


def test_load_from_disk_ticker_exists(loader, mock_response):
    """Test that _load_from_disk returns DataFrame when file exists."""
    # Create test data
    df = pd.DataFrame(mock_response["results"])
    df = df.rename(columns={
        "v": "volume",
        "o": "open",
        "c": "close",
        "h": "high",
        "l": "low",
        "t": "timestamp"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    # Save to disk
    data_path = os.path.join("test_data", "AAPL.parquet")
    os.makedirs("test_data", exist_ok=True)
    df.to_parquet(data_path)
    
    # Test loading
    loaded_df = loader._load_from_disk("AAPL")
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 2
    assert list(loaded_df.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    # Clean up
    os.remove(data_path)
    os.rmdir("test_data")


def test_load_from_disk_data_path_doesnt_exist(loader, caplog):
    """Test that _load_from_disk logs warning and returns None when file doesn't exist."""
    result = loader._load_from_disk("NONEXISTENT")
    assert result is None
    assert "Error loading data for NONEXISTENT" in caplog.text


def test_load_from_disk_data_is_malformed(loader, caplog):
    """Test that _load_from_disk logs warning and returns None when data is malformed."""
    # Create malformed parquet file
    data_path = os.path.join("test_data", "MALFORMED.parquet")
    os.makedirs("test_data", exist_ok=True)
    with open(data_path, 'w') as f:
        f.write("not a parquet file")
    
    # Test loading
    result = loader._load_from_disk("MALFORMED")
    assert result is None
    assert "Error loading data for MALFORMED" in caplog.text
    
    # Clean up
    os.remove(data_path)
    os.rmdir("test_data")


def test_save_to_disk_invalid_ticker(loader):
    """Test that _save_to_disk raises error for invalid ticker characters."""
    df = pd.DataFrame({'test': [1, 2, 3]})
    with pytest.raises(ValueError):
        loader._save_to_disk("INVALID/TICKER", df)


def test_save_to_disk_valid_dataframe(loader):
    """Test that _save_to_disk correctly saves and can be loaded back."""
    # Create test data
    df = pd.DataFrame({
        'open': [100.0, 102.0],
        'high': [105.0, 107.0],
        'low': [95.0, 97.0],
        'close': [102.0, 104.0],
        'volume': [1000, 1100]
    }, index=pd.date_range('2021-01-01', '2021-01-02'))
    
    # Save to disk
    loader._save_to_disk("AAPL", df)
    
    # Verify file exists
    data_path = os.path.join("test_data", "AAPL.parquet")
    assert os.path.exists(data_path)
    
    # Load and verify data
    loaded_df = pd.read_parquet(data_path)
    
    # Reset index names to None for comparison
    df.index.name = None
    loaded_df.index.name = None
    
    # Compare data values only, ignoring index type
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        loaded_df.reset_index(drop=True)
    )
    
    # Clean up
    os.remove(data_path)
    os.rmdir("test_data")


def test_fetch_historical_data_pagination(loader):
    """Test that fetch_historical_data handles pagination correctly."""
    # Mock responses for pagination
    first_response = {
        "results": [
            {
                "t": 1609459200000,  # 2021-01-01
                "o": 100.0,
                "h": 105.0,
                "l": 95.0,
                "c": 102.0,
                "v": 1000
            }
        ],
        "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2021-01-01/2021-01-02?cursor=next"
    }
    
    second_response = {
        "results": [
            {
                "t": 1609545600000,  # 2021-01-02
                "o": 102.0,
                "h": 107.0,
                "l": 97.0,
                "c": 104.0,
                "v": 1100
            }
        ]
    }
    
    with patch('requests.Session.get') as mock_get:
        # Set up mock to return different responses
        mock_get.side_effect = [
            MagicMock(
                json=lambda: first_response,
                raise_for_status=MagicMock()
            ),
            MagicMock(
                json=lambda: second_response,
                raise_for_status=MagicMock()
            )
        ]
        
        # Fetch data
        df = loader.fetch_historical_data("AAPL", from_date="2021-01-01", to_date="2021-01-02")
        
        # Verify data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        
        # Verify cache
        assert "AAPL" in loader._data_cache
        assert isinstance(loader._data_cache["AAPL"], pd.DataFrame)
        
        # Verify disk storage
        data_path = os.path.join("test_data", "AAPL.parquet")
        assert os.path.exists(data_path)
        
        # Clean up
        os.remove(data_path)
        os.rmdir("test_data")


def test_validate_and_convert_dates_valid_ymd(loader):
    """Test that _validate_and_convert_dates handles YYYY-MM-DD format correctly."""
    from_date, to_date = loader._validate_and_convert_dates("2021-01-01", "2021-01-02")
    assert from_date == "1609459200000"  # 2021-01-01 00:00:00 UTC in ms
    assert to_date == "1609545600000"    # 2021-01-02 00:00:00 UTC in ms


def test_validate_and_convert_dates_valid_ms(loader):
    """Test that _validate_and_convert_dates handles millisecond timestamps correctly."""
    from_date, to_date = loader._validate_and_convert_dates("1609459200000", "1609545600000")
    assert from_date == "1609459200000"
    assert to_date == "1609545600000"


def test_validate_and_convert_dates_invalid_ymd(loader):
    """Test that _validate_and_convert_dates raises error for invalid YYYY-MM-DD format."""
    with pytest.raises(ValueError):
        loader._validate_and_convert_dates("2021-13-01", "2021-01-02")


def test_validate_and_convert_dates_invalid_ms(loader):
    """Test that _validate_and_convert_dates raises error for invalid millisecond timestamp."""
    with pytest.raises(ValueError):
        loader._validate_and_convert_dates("not_a_timestamp", "1609545600000")


def test_validate_and_convert_dates_wrong_order(loader):
    """Test that _validate_and_convert_dates raises error when to_date is before from_date."""
    with pytest.raises(ValueError):
        loader._validate_and_convert_dates("1609545600000", "1609459200000")


def test_validate_and_convert_dates_defaults(loader):
    """Test that _validate_and_convert_dates sets proper defaults."""
    from_date, to_date = loader._validate_and_convert_dates()
    assert from_date.isdigit()  # Should be millisecond timestamp
    assert to_date.isdigit()    # Should be millisecond timestamp
    assert int(to_date) > int(from_date)


def test_fetch_historical_data_malformed_to_date_millisecond_timestamp(loader):
    """Test that fetch_historical_data raises error for malformed to_date with millisecond timestamp."""
    with pytest.raises(ValueError):
        loader.fetch_historical_data("AAPL", to_date="invalid_timestamp")


def test_fetch_historical_data_malformed_from_date_millisecond_timestamp(loader):
    """Test that fetch_historical_data raises error for malformed from_date with millisecond timestamp."""
    with pytest.raises(ValueError):
        loader.fetch_historical_data("AAPL", from_date="invalid_timestamp")


def test_fetch_historical_data_to_is_after_from_date_millisecond_timestamp(loader):
    """Test that fetch_historical_data raises error when to_date is after from_date with millisecond timestamps."""
    with pytest.raises(ValueError):
        loader.fetch_historical_data(
            "AAPL",
            from_date="1609545600000",  # 2021-01-02
            to_date="1609459200000"     # 2021-01-01
        )


def test_fetch_historical_data_works_millisecond_timestamp(loader, mock_response):
    """Test that fetch_historical_data returns data for the entire period using millisecond timestamps."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        df = loader.fetch_historical_data(
            "AAPL",
            from_date="1609459200000",  # 2021-01-01
            to_date="1609545600000"     # 2021-01-02
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']


def test_fetch_historical_data_adds_to_cache_millisecond_timestamp(loader, mock_response):
    """Test that fetch_historical_data adds data to in-memory cache using millisecond timestamps."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        loader.fetch_historical_data(
            "AAPL",
            from_date="1609459200000",  # 2021-01-01
            to_date="1609545600000"     # 2021-01-02
        )
        assert "AAPL" in loader._data_cache
        assert isinstance(loader._data_cache["AAPL"], pd.DataFrame)


def test_fetch_historical_data_adds_saves_to_disk_millisecond_timestamp(loader, mock_response):
    """Test that fetch_historical_data saves data to disk using millisecond timestamps."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        loader.fetch_historical_data(
            "AAPL",
            from_date="1609459200000",  # 2021-01-01
            to_date="1609545600000"     # 2021-01-02
        )
        data_path = os.path.join("test_data", "AAPL.parquet")
        assert os.path.exists(data_path)
        
        # Clean up
        os.remove(data_path)
        os.rmdir("test_data")


def test_api_get_valid_response_millisecond_timestamp(loader, mock_response):
    """Test that _api_get returns valid JSON response with millisecond timestamps."""
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        response = loader._api_get("test_url")
        assert response == mock_response


@freeze_time("2021-01-01 10:30:00", tz_offset=0)
def test_fetch_partial_day_data_up_to_now(loader):
    """Test that _fetch_partial_day_data returns data up until now."""
    # Mock current time in US/Eastern
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    now_ms = int(now.timestamp() * 1000)
    
    # Mock response with data up to now
    mock_response = {
        "results": [
            {
                "t": now_ms - (60 * 1000),  # 1 minute ago
                "o": 100.0,
                "h": 105.0,
                "l": 95.0,
                "c": 102.0,
                "v": 1000
            }
        ]
    }
    
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        result = loader._fetch_partial_day_data("AAPL")
        assert result is not None
        assert result["open"] == 100.0
        assert result["high"] == 105.0
        assert result["low"] == 95.0
        assert result["close"] == 102.0
        assert result["volume"] == 1000.0


@freeze_time("2021-01-01 08:30:00", tz_offset=0)  # 8:30 AM ET
def test_fetch_partial_day_data_only_from_day(loader):
    """Test that _fetch_partial_day_data only includes data from trading hours."""
    # Mock current time in US/Eastern
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    now_ms = int(now.timestamp() * 1000)
    
    # Mock response with data outside trading hours
    mock_response = {
        "results": [
            {
                "t": now_ms - (60 * 1000),  # 1 minute ago
                "o": 100.0,
                "h": 105.0,
                "l": 95.0,
                "c": 102.0,
                "v": 1000
            }
        ]
    }
    
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        result = loader._fetch_partial_day_data("AAPL")
        assert result is None


@freeze_time("2021-01-01 10:30:00", tz_offset=0)
def test_fetch_partial_day_data_builds_ohlcv(loader):
    """Test that _fetch_partial_day_data correctly builds OHLCV data."""
    # Mock current time in US/Eastern during trading hours
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est).replace(hour=10, minute=30)  # 10:30 AM ET
    now_ms = int(now.timestamp() * 1000)
    
    # Mock response with multiple bars
    mock_response = {
        "results": [
            {
                "t": now_ms - (60 * 1000),  # 1 minute ago
                "o": 100.0,
                "h": 105.0,
                "l": 95.0,
                "c": 102.0,
                "v": 1000
            },
            {
                "t": now_ms - (30 * 1000),  # 30 seconds ago
                "o": 102.0,
                "h": 107.0,
                "l": 97.0,
                "c": 104.0,
                "v": 1100
            }
        ]
    }
    
    with patch('requests.Session.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = MagicMock()
        
        result = loader._fetch_partial_day_data("AAPL")
        assert result is not None
        assert result["open"] == 100.0  # First bar's open
        assert result["high"] == 107.0  # Highest high
        assert result["low"] == 95.0    # Lowest low
        assert result["close"] == 104.0  # Last bar's close
        assert result["volume"] == 2100.0  # Sum of volumes


def test_fetch_partial_day_data_exception(loader, caplog):
    """Test that _fetch_partial_day_data handles exceptions correctly."""
    with patch('requests.Session.get') as mock_get:
        mock_get.side_effect = Exception("Test error")
        
        result = loader._fetch_partial_day_data("AAPL")
        assert result is None
        assert "Error fetching partial day data for AAPL" in caplog.text


def test_load_ticker_from_cache(loader):
    """Test that load_ticker returns data from cache if available."""
    # Create test data
    df = pd.DataFrame({
        'open': [100.0, 102.0],
        'high': [105.0, 107.0],
        'low': [95.0, 97.0],
        'close': [102.0, 104.0],
        'volume': [1000, 1100]
    }, index=pd.date_range('2021-01-01', '2021-01-02'))
    
    # Add to cache
    loader._data_cache['AAPL'] = df
    
    # Test loading from cache
    result = loader.load_ticker('AAPL')
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_load_ticker_from_disk(loader):
    """Test that load_ticker loads from disk and adds to cache."""
    # Create test data
    df = pd.DataFrame({
        'open': [100.0, 102.0],
        'high': [105.0, 107.0],
        'low': [95.0, 97.0],
        'close': [102.0, 104.0],
        'volume': [1000, 1100]
    }, index=pd.date_range('2021-01-01', '2021-01-02'))
    
    # Save to disk
    data_path = os.path.join("test_data", "AAPL.parquet")
    os.makedirs("test_data", exist_ok=True)
    df.to_parquet(data_path)
    
    # Test loading from disk
    result = loader.load_ticker('AAPL')
    assert isinstance(result, pd.DataFrame)
    
    # Compare data values only, ignoring index type
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        result.reset_index(drop=True)
    )
    
    # Verify it was added to cache
    assert 'AAPL' in loader._data_cache
    pd.testing.assert_frame_equal(
        loader._data_cache['AAPL'].reset_index(drop=True),
        df.reset_index(drop=True)
    )
    
    # Clean up
    os.remove(data_path)
    os.rmdir("test_data")


def test_load_ticker_not_found(loader):
    """Test that load_ticker returns None when ticker is not found."""
    # Mock fetch_historical_data to return None
    with patch.object(loader, 'fetch_historical_data', return_value=None):
        result = loader.load_ticker('NONEXISTENT')
        assert result is None
        assert 'NONEXISTENT' not in loader._data_cache 
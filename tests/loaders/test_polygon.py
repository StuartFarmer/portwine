import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import requests
from portwine.loaders.polygon import PolygonMarketDataLoader


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
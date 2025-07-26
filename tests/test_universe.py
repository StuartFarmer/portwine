"""
Tests for the Universe class.
"""

import pytest
import pandas as pd
from datetime import date, datetime
import tempfile
import os
from portwine.universe import Universe


class TestUniverse:
    """Test cases for Universe class."""
    
    def create_temp_csv(self, data):
        """Helper to create temporary CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("date,basket\n")
        for row in data:
            temp_file.write(f"{row[0]},{row[1]}\n")
        temp_file.close()
        return temp_file.name
    
    def test_basic_functionality(self):
        """Test basic universe functionality."""
        data = [
            ["2020-01-01", "AAPL,GOOGL,MSFT"],
            ["2020-02-01", "AAPL,GOOGL,MSFT,AMZN"],
            ["2020-03-01", "AAPL,MSFT,AMZN"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Test exact date matches
            assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL", "MSFT"]
            assert universe.get_constituents("2020-02-01") == ["AAPL", "GOOGL", "MSFT", "AMZN"]
            assert universe.get_constituents("2020-03-01") == ["AAPL", "MSFT", "AMZN"]
            
            # Test dates between snapshots
            assert universe.get_constituents("2020-01-15") == ["AAPL", "GOOGL", "MSFT"]
            assert universe.get_constituents("2020-02-15") == ["AAPL", "GOOGL", "MSFT", "AMZN"]
            
            # Test dates after last snapshot
            assert universe.get_constituents("2020-04-01") == ["AAPL", "MSFT", "AMZN"]
            
        finally:
            os.unlink(csv_path)
    
    def test_before_first_date(self):
        """Test behavior when date is before first snapshot."""
        data = [
            ["2020-02-01", "AAPL,GOOGL"],
            ["2020-03-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Date before first snapshot should return empty list
            assert universe.get_constituents("2020-01-01") == []
            assert universe.get_constituents("2019-12-31") == []
            
        finally:
            os.unlink(csv_path)
    
    def test_datetime_objects(self):
        """Test that datetime objects work correctly."""
        data = [
            ["2020-01-01", "AAPL,GOOGL"],
            ["2020-02-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Test with datetime objects
            dt1 = datetime(2020, 1, 15, 10, 30, 0)  # 10:30 AM
            dt2 = datetime(2020, 2, 15, 15, 45, 30)  # 3:45 PM
            
            assert universe.get_constituents(dt1) == ["AAPL", "GOOGL"]
            assert universe.get_constituents(dt2) == ["AAPL", "GOOGL", "MSFT"]
            
        finally:
            os.unlink(csv_path)
    
    def test_pandas_timestamp(self):
        """Test that pandas Timestamp objects work correctly."""
        data = [
            ["2020-01-01", "AAPL,GOOGL"],
            ["2020-02-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Test with pandas Timestamp
            ts1 = pd.Timestamp("2020-01-15 09:30:00")
            ts2 = pd.Timestamp("2020-02-15 16:00:00")
            
            assert universe.get_constituents(ts1) == ["AAPL", "GOOGL"]
            assert universe.get_constituents(ts2) == ["AAPL", "GOOGL", "MSFT"]
            
        finally:
            os.unlink(csv_path)
    
    def test_single_snapshot(self):
        """Test universe with only one snapshot."""
        data = [
            ["2020-01-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Before snapshot
            assert universe.get_constituents("2019-12-31") == []
            
            # At snapshot
            assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL", "MSFT"]
            
            # After snapshot
            assert universe.get_constituents("2020-12-31") == ["AAPL", "GOOGL", "MSFT"]
            
        finally:
            os.unlink(csv_path)
    
    def test_empty_basket(self):
        """Test handling of empty baskets."""
        data = [
            ["2020-01-01", ""],
            ["2020-02-01", "AAPL,GOOGL"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Empty basket should return empty list
            assert universe.get_constituents("2020-01-01") == [""]
            assert universe.get_constituents("2020-01-15") == [""]
            
            # Non-empty basket
            assert universe.get_constituents("2020-02-01") == ["AAPL", "GOOGL"]
            
        finally:
            os.unlink(csv_path)
    
    def test_single_ticker(self):
        """Test universe with single ticker in basket."""
        data = [
            ["2020-01-01", "AAPL"],
            ["2020-02-01", "GOOGL"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            assert universe.get_constituents("2020-01-01") == ["AAPL"]
            assert universe.get_constituents("2020-02-01") == ["GOOGL"]
            
        finally:
            os.unlink(csv_path)
    
    def test_duplicate_dates(self):
        """Test behavior with duplicate dates (should use last one)."""
        data = [
            ["2020-01-01", "AAPL,GOOGL"],
            ["2020-01-01", "MSFT,AMZN"],  # Duplicate date
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Should use the last entry for the date
            assert universe.get_constituents("2020-01-01") == ["MSFT", "AMZN"]
            
        finally:
            os.unlink(csv_path)
    
    def test_date_format_variations(self):
        """Test different date formats in CSV."""
        data = [
            ["2020-01-01 00:00:00", "AAPL,GOOGL"],
            ["2020-02-01 12:30:45", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Should normalize to date only
            assert universe.get_constituents("2020-01-15") == ["AAPL", "GOOGL"]
            assert universe.get_constituents("2020-02-15") == ["AAPL", "GOOGL", "MSFT"]
            
        finally:
            os.unlink(csv_path)
    
    def test_binary_search_edge_cases(self):
        """Test binary search edge cases."""
        data = [
            ["2020-01-01", "A"],
            ["2020-02-01", "B"],
            ["2020-03-01", "C"],
            ["2020-04-01", "D"],
            ["2020-05-01", "E"],
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Test exact matches
            assert universe.get_constituents("2020-01-01") == ["A"]
            assert universe.get_constituents("2020-03-01") == ["C"]
            assert universe.get_constituents("2020-05-01") == ["E"]
            
            # Test between dates
            assert universe.get_constituents("2020-01-15") == ["A"]
            assert universe.get_constituents("2020-02-15") == ["B"]
            assert universe.get_constituents("2020-04-15") == ["D"]
            
            # Test after last date
            assert universe.get_constituents("2020-06-01") == ["E"]
            
        finally:
            os.unlink(csv_path)
    
    def test_malformed_csv(self):
        """Test handling of malformed CSV."""
        # Missing basket column
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("date\n2020-01-01\n")
        temp_file.close()
        
        try:
            with pytest.raises(KeyError):
                Universe(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Universe("nonexistent_file.csv")
    
    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("date,basket\ninvalid_date,AAPL\n")
        temp_file.close()
        
        try:
            with pytest.raises(ValueError):
                Universe(temp_file.name)
        finally:
            os.unlink(temp_file.name)
    
    def test_whitespace_in_basket(self):
        """Test handling of whitespace in basket."""
        data = [
            ["2020-01-01", " AAPL , GOOGL , MSFT "],  # Extra whitespace
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            # Should preserve whitespace as-is
            assert universe.get_constituents("2020-01-01") == [" AAPL ", " GOOGL ", " MSFT "]
            
        finally:
            os.unlink(csv_path)
    
    def test_unicode_characters(self):
        """Test handling of unicode characters in tickers."""
        data = [
            ["2020-01-01", "AAPL,GOOGL,BRK.B"],  # BRK.B has special character
        ]
        
        csv_path = self.create_temp_csv(data)
        
        try:
            universe = Universe(csv_path)
            
            assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL", "BRK.B"]
            
        finally:
            os.unlink(csv_path)
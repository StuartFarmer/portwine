"""
Tests for the Universe class.
"""

import pytest
from datetime import date, datetime
import os
from portwine.universe import Universe


class TestUniverse:
    """Test cases for Universe class."""
    
    def setup_method(self):
        """Create test files before each test."""
        self.test_files = []
    
    def teardown_method(self):
        """Clean up test files after each test."""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def create_test_csv(self, data, filename):
        """Helper to create test CSV file."""
        file_path = f"tests/test_data/{filename}"
        os.makedirs("tests/test_data", exist_ok=True)
        
        with open(file_path, 'w') as f:
            for row in data:
                f.write(f"{row[0]},{row[1]}\n")
        
        self.test_files.append(file_path)
        return file_path
    
    def test_basic_functionality(self):
        """Test basic universe functionality."""
        data = [
            ["2020-01-01", "AAPL,GOOGL,MSFT"],
            ["2020-02-01", "AAPL,GOOGL,MSFT,AMZN"],
            ["2020-03-01", "AAPL,MSFT,AMZN"],
        ]
        
        csv_path = self.create_test_csv(data, "basic_universe.csv")
        
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
    
    def test_before_first_date(self):
        """Test behavior when date is before first snapshot."""
        data = [
            ["2020-02-01", "AAPL,GOOGL"],
            ["2020-03-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_test_csv(data, "before_first_date.csv")
        
        universe = Universe(csv_path)
        
        # Date before first snapshot should return empty list
        assert universe.get_constituents("2020-01-01") == []
        assert universe.get_constituents("2019-12-31") == []
    
    def test_datetime_objects(self):
        """Test that datetime objects work correctly."""
        data = [
            ["2020-01-01", "AAPL,GOOGL"],
            ["2020-02-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_test_csv(data, "datetime_objects.csv")
        
        universe = Universe(csv_path)
        
        # Test with datetime objects
        dt1 = datetime(2020, 1, 15, 10, 30, 0)  # 10:30 AM
        dt2 = datetime(2020, 2, 15, 15, 45, 30)  # 3:45 PM
        
        assert universe.get_constituents(dt1) == ["AAPL", "GOOGL"]
        assert universe.get_constituents(dt2) == ["AAPL", "GOOGL", "MSFT"]
    
    def test_single_snapshot(self):
        """Test universe with only one snapshot."""
        data = [
            ["2020-01-01", "AAPL,GOOGL,MSFT"],
        ]
        
        csv_path = self.create_test_csv(data, "single_snapshot.csv")
        
        universe = Universe(csv_path)
        
        # Before snapshot
        assert universe.get_constituents("2019-12-31") == []
        
        # At snapshot
        assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL", "MSFT"]
        
        # After snapshot
        assert universe.get_constituents("2020-12-31") == ["AAPL", "GOOGL", "MSFT"]
    
    def test_empty_basket(self):
        """Test handling of empty baskets."""
        data = [
            ["2020-01-01", ""],
            ["2020-02-01", "AAPL,GOOGL"],
        ]
        
        csv_path = self.create_test_csv(data, "empty_basket.csv")
        
        universe = Universe(csv_path)
        
        # Empty basket should return empty list
        assert universe.get_constituents("2020-01-01") == []
        assert universe.get_constituents("2020-01-15") == []
        
        # Non-empty basket
        assert universe.get_constituents("2020-02-01") == ["AAPL", "GOOGL"]
    
    def test_single_ticker(self):
        """Test universe with single ticker in basket."""
        data = [
            ["2020-01-01", "AAPL"],
            ["2020-02-01", "GOOGL"],
        ]
        
        csv_path = self.create_test_csv(data, "single_ticker.csv")
        
        universe = Universe(csv_path)
        
        assert universe.get_constituents("2020-01-01") == ["AAPL"]
        assert universe.get_constituents("2020-02-01") == ["GOOGL"]
    
    def test_duplicate_dates(self):
        """Test behavior with duplicate dates (should use last one)."""
        data = [
            ["2020-01-01", "AAPL,GOOGL"],
            ["2020-01-01", "MSFT,AMZN"],  # Duplicate date
        ]
        
        csv_path = self.create_test_csv(data, "duplicate_dates.csv")
        
        universe = Universe(csv_path)
        
        # Should use the last entry for the date
        assert universe.get_constituents("2020-01-01") == ["MSFT", "AMZN"]
    
    def test_binary_search_edge_cases(self):
        """Test binary search edge cases."""
        data = [
            ["2020-01-01", "A"],
            ["2020-02-01", "B"],
            ["2020-03-01", "C"],
            ["2020-04-01", "D"],
            ["2020-05-01", "E"],
        ]
        
        csv_path = self.create_test_csv(data, "binary_search_edge_cases.csv")
        
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
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Universe("nonexistent_file.csv")
    
    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        file_path = "tests/test_data/invalid_date.csv"
        os.makedirs("tests/test_data", exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write("invalid_date,AAPL\n")
        
        self.test_files.append(file_path)
        
        universe = Universe(file_path)
        # Should skip invalid dates and return empty
        assert universe.get_constituents("2020-01-01") == []
    
    def test_whitespace_in_basket(self):
        """Test handling of whitespace in basket."""
        data = [
            ["2020-01-01", " AAPL , GOOGL , MSFT "],  # Extra whitespace
        ]
        
        csv_path = self.create_test_csv(data, "whitespace_in_basket.csv")
        
        universe = Universe(csv_path)
        
        # Should strip whitespace
        assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL", "MSFT"]
    
    def test_unicode_characters(self):
        """Test handling of unicode characters in tickers."""
        data = [
            ["2020-01-01", "AAPL,GOOGL,BRK.B"],  # BRK.B has special character
        ]
        
        csv_path = self.create_test_csv(data, "unicode_characters.csv")
        
        universe = Universe(csv_path)
        
        assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL", "BRK.B"]
    
    def test_comments_and_empty_lines(self):
        """Test handling of comments and empty lines."""
        file_path = "tests/test_data/comments.csv"
        os.makedirs("tests/test_data", exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write("# This is a comment\n")
            f.write("\n")  # Empty line
            f.write("2020-01-01,AAPL,GOOGL\n")
            f.write("# Another comment\n")
            f.write("2020-02-01,MSFT,AMZN\n")
        
        self.test_files.append(file_path)
        
        universe = Universe(file_path)
        
        assert universe.get_constituents("2020-01-01") == ["AAPL", "GOOGL"]
        assert universe.get_constituents("2020-02-01") == ["MSFT", "AMZN"]
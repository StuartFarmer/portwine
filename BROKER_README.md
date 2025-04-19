# Broker and Execution Refactoring

This document explains the refactoring of the execution system in Portwine, which separates broker functionality from the execution logic.

## Overview

The refactoring introduces a clean separation of concerns:

1. **Single ExecutionBase Class** - Handles core execution logic for all trading
2. **Multiple Broker Implementations** - Handle broker-specific functionality through the BrokerBase interface
3. **Utility Functions** - Common utilities extracted to their own module

This simplified structure provides several benefits:
- Better testability with mock broker implementations
- Clearer responsibility boundaries
- Easier to add new broker implementations (without creating new execution classes)
- Reduced duplication of code

## Components

### Broker Module

The `broker.py` module introduces these core classes:

1. **AccountInfo** - Container for account information
   - Cash balance
   - Portfolio value
   - Positions

2. **Position** - Container for position information
   - Symbol
   - Quantity
   - Market value
   - Average entry price
   - Unrealized P&L

3. **Order** - Container for order information
   - Symbol
   - Quantity (positive for buy, negative for sell)
   - Order type (market, limit, etc.)
   - Limit price (if applicable)
   - Time in force

4. **BrokerBase** - Abstract base class for broker implementations with these key methods:
   - `check_market_status()` - Check if market is open
   - `get_account_info()` - Get current account information
   - `execute_order()` - Execute a trade order
   - `get_position()` - Get information about a specific position
   - `get_portfolio_weights()` - Calculate current portfolio weights

### Execution Utilities

The `execution_utils.py` module contains common functions used by the execution system:

1. **create_bar_dict()** - Create a dictionary of bar data from a DataFrame
2. **calculate_position_changes()** - Calculate position changes needed to reach target positions
3. **generate_orders()** - Generate orders from position changes

### ExecutionBase

The `ExecutionBase` class has been updated to:
- Work with any broker implementation that follows the BrokerBase interface
- Use utility functions for common operations
- Provide a clean, maintainable interface regardless of the broker used

## Broker Implementations

The system includes these broker implementations:

1. **AlpacaBroker** - Implementation for Alpaca trading API
2. **MockBroker** - Implementation for testing without real trading

## Example Usage

The `example_broker_execution.py` script demonstrates how to use the components:

```python
# Create a strategy
strategy = SimpleMovingAverageStrategy(tickers=["AAPL", "MSFT"])

# OPTION 1: Using the mock broker for testing
mock_broker = MockBroker(initial_cash=100000.0)
execution_with_mock = ExecutionBase(
    strategy=strategy,
    market_data_loader=data_loader,
    broker=mock_broker
)

# OPTION 2: Using the Alpaca broker for real trading
alpaca_broker = AlpacaBroker(
    api_key="your_key",
    api_secret="your_secret",
    paper_trading=True  # Set to False for live trading
)
execution_with_alpaca = ExecutionBase(
    strategy=strategy,
    market_data_loader=data_loader,
    broker=alpaca_broker
)

# Run a trading step - same code regardless of broker implementation
results = execution_with_alpaca.step()
```

## Adding New Broker Implementations

To add a new broker implementation:

1. Create a new class that inherits from `BrokerBase`
2. Implement all required abstract methods
3. Use it with ExecutionBase

Example:

```python
class MyBroker(BrokerBase):
    def __init__(self, credentials):
        self.api = MyBrokerAPI(credentials)
        
    def check_market_status(self) -> bool:
        return self.api.is_market_open()
        
    def get_account_info(self) -> Dict[str, Any]:
        account = self.api.get_account()
        return {
            'cash': account.cash,
            'portfolio_value': account.equity,
            'positions': self._convert_positions(account.positions)
        }
        
    def execute_order(self, symbol, qty, order_type="market", **kwargs) -> bool:
        try:
            self.api.place_order(symbol, qty, order_type, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return False
            
    # Implement other required methods...

# Then use it with ExecutionBase:
my_broker = MyBroker(credentials=my_credentials)
execution = ExecutionBase(
    strategy=strategy,
    market_data_loader=data_loader,
    broker=my_broker
)
```

## Architecture Benefits

This architecture provides a plug-and-play approach to broker implementations:

1. **Swappable Brokers** - Easily switch between paper trading, live trading, or mock trading
2. **Consistent Interface** - Same ExecutionBase class works with any broker implementation
3. **Strategy Independence** - Strategies don't need to know about the broker implementation
4. **Testable** - Easy to create mock broker implementations for testing
5. **Extensible** - Add new broker implementations without changing existing code

## Future Improvements

Planned improvements to the broker and execution system:

1. **Support for Different Asset Types** - Stocks, options, futures, crypto
2. **Advanced Order Types** - Bracket orders, OCO orders, etc.
3. **Order Book Integration** - For better order execution at optimal prices
4. **Risk Management** - Position size limits, drawdown protection, etc.
5. **Multi-Account Support** - Managing multiple trading accounts 
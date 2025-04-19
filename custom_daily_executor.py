#!/usr/bin/env python
"""
Custom DailyExecutor that correctly handles initialization of the AlpacaExecution class.
"""

import logging
from typing import Dict, Any

from portwine.utils.daily_executor import DailyExecutor

logger = logging.getLogger(__name__)

class CustomDailyExecutor(DailyExecutor):
    """
    Custom DailyExecutor that properly initializes components with the correct parameters.
    
    This subclass specifically addresses issues with the AlpacaExecution class initialization
    by ensuring that market_data_loader is correctly passed as a parameter.
    """
    
    def initialize(self) -> None:
        """
        Initialize all components based on configuration.
        
        This includes:
        - Data loader
        - Strategy
        - Execution system
        
        This custom version ensures that market_data_loader is correctly passed to the execution_complex system.
        """
        if self.initialized:
            logger.warning("Components already initialized")
            return
            
        logger.info("Initializing components...")
        
        # Initialize data loader
        if "data_loader" in self.config:
            data_loader_config = self.config["data_loader"]
            data_loader_class = self._import_class(data_loader_config["class"])
            data_loader_params = data_loader_config.get("params", {})
            self.data_loader = data_loader_class(**data_loader_params)
            logger.info(f"Initialized data loader: {data_loader_class.__name__}")
        
        # Initialize strategy
        if "strategy" in self.config:
            strategy_config = self.config["strategy"]
            strategy_class = self._import_class(strategy_config["class"])
            strategy_params = strategy_config.get("params", {})
            
            # If tickers are specified separately, add them to params
            if "tickers" in strategy_config:
                strategy_params["tickers"] = strategy_config["tickers"]
                
            self.strategy = strategy_class(**strategy_params)
            logger.info(f"Initialized strategy: {strategy_class.__name__}")
        
        # Initialize execution_complex system
        if "execution_complex" in self.config:
            execution_config = self.config["execution_complex"]
            execution_class = self._import_class(execution_config["class"])
            execution_params = execution_config.get("params", {})
            
            # Add strategy if not already provided
            if self.strategy is not None and "strategy" not in execution_params:
                execution_params["strategy"] = self.strategy
                
            # Add market_data_loader if not already provided (this is the key fix)
            if self.data_loader is not None and "market_data_loader" not in execution_params:
                execution_params["market_data_loader"] = self.data_loader
                
            self.executor = execution_class(**execution_params)
            logger.info(f"Initialized execution_complex system: {execution_class.__name__}")
        
        self.initialized = True
        logger.info("All components initialized successfully") 
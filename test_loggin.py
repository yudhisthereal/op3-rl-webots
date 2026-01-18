#!/usr/bin/env python3
"""
Test script for logging system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from logging_utils import (
    log, log_info, log_warning, log_error, log_success,
    log_debug, log_data, log_exception, log_section,
    start_timer, stop_timer, LogFunction, logged_function,
    setup_logging
)

def test_basic_logging():
    """Test basic logging functions."""
    log_section("Test", "BASIC LOGGING TESTS")
    
    log_info("Test", "This is an info message")
    log_warning("Test", "This is a warning message")
    log_error("Test", "This is an error message")
    log_success("Test", "This is a success message")
    log_debug("Test", "This is a debug message (might be hidden)")
    
    # Test data logging
    sample_data = {
        "name": "Test Agent",
        "fitness": 0.85,
        "generation": 10,
        "hyperparameters": {"lr": 0.001, "gamma": 0.99}
    }
    log_data("Test", "Sample Agent Data", sample_data)
    
    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        log_exception("Test", e, "Testing exception logging")

def test_timing():
    """Test timing functions."""
    log_section("Test", "TIMING TESTS")
    
    start_timer("test_operation")
    
    # Simulate some work
    import time
    time.sleep(0.1)
    
    elapsed = stop_timer("test_operation")
    log_info("Test", f"Operation took {elapsed:.4f} seconds")

@logged_function(tag="Test")
def test_logged_function(x: int, y: int = 10) -> int:
    """Test function with logging decorator."""
    log_info("TestFunction", f"Processing x={x}, y={y}")
    result = x * y
    log_data("TestFunction", "Intermediate result", result)
    return result + 5

def test_context_manager():
    """Test context manager logging."""
    with LogFunction("Test", "context_managed_function", args=(1, 2), kwargs={"z": 3}):
        log_info("TestContext", "Inside context manager")
        result = 1 + 2 + 3
        log_data("TestContext", "Context result", result)
        return result

def main():
    """Run all logging tests."""
    # Setup logging with debug enabled
    logger = setup_logging()
    logger.config.set("show_debug", True)
    
    log_section("Main", "STARTING LOGGING TESTS")
    
    test_basic_logging()
    test_timing()
    
    # Test logged function
    result = test_logged_function(5, y=3)
    log_data("Main", "Logged function result", result)
    
    # Test context manager
    result = test_context_manager()
    log_data("Main", "Context manager result", result)
    
    # Test with exception
    try:
        with LogFunction("Test", "failing_function"):
            raise RuntimeError("Intentional failure for testing")
    except:
        log_warning("Main", "Caught expected exception")
    
    log_success("Main", "All logging tests completed successfully")

if __name__ == "__main__":
    main()
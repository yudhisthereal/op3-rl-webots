"""
Logging Utility for OP3 Robot Training System.

Provides a unified logging system with color coding and configuration support.
"""

import os
import sys
import json
import time
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import traceback

# =============================================================================
# Color Codes for Terminal Output
# =============================================================================

class LogColors:
    """ANSI color codes for terminal output."""
    # Styles
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Colors - Regular
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Colors - Bright/Bold
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Backgrounds
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Special Colors (for specific log types)
    ERROR = '\033[38;5;203m'  # Bold peach/orange-red
    WARNING = '\033[38;5;227m'  # Bold yellow
    INFO = '\033[38;5;117m'  # Washed out blue
    SUCCESS = '\033[38;5;83m'  # Bright green
    DEBUG = '\033[38;5;141m'  # Purple
    DATA = '\033[38;5;215m'  # Orange
    TIMING = '\033[38;5;171m'  # Pink


# =============================================================================
# Log Configuration
# =============================================================================

class LogConfig:
    """Manages logging configuration."""
    
    DEFAULT_CONFIG = {
        "show_info": True,
        "show_warning": True,
        "show_error": True,
        "show_debug": False,
        "show_success": True,
        "show_data": True,
        "show_timing": True,
        "use_colors": True,
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "log_file": None,
        "max_message_length": 2000,
        "indent_size": 2,
        "caller_max_length": 20,
        "log_to_file": False,
        "file_mode": "a",  # 'a' for append, 'w' for overwrite
        "file_encoding": "utf-8",
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize log configuration.
        
        Args:
            config_path: Path to config file (default: project_root/log_config.json)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_path = config_path
        
        if config_path is None:
            # Try to find config in project root
            project_root = Path(__file__).parent
            self.config_path = project_root / "log_config.json"
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Update with file config (preserve defaults for missing keys)
                for key, value in file_config.items():
                    if key in self.config:
                        self.config[key] = value
                
                self._log_config_load_success()
            except Exception as e:
                self._log_config_load_error(e)
    
    def _log_config_load_success(self):
        """Internal method to log successful config load."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.INFO}📋 Loaded log config from: {self.config_path}{LogColors.RESET}")
        else:
            print(f"📋 Loaded log config from: {self.config_path}")
    
    def _log_config_load_error(self, e: Exception):
        """Internal method to log config load error."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.WARNING}⚠️ Failed to load log config: {e}, using defaults{LogColors.RESET}")
        else:
            print(f"⚠️ Failed to load log config: {e}, using defaults")
    
    def save_config(self):
        """Save configuration to file."""
        if self.config_path:
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                return True
            except Exception as e:
                self._log_config_save_error(e)
                return False
        return False
    
    def _log_config_save_error(self, e: Exception):
        """Internal method to log config save error."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.WARNING}⚠️ Failed to save log config: {e}{LogColors.RESET}")
        else:
            print(f"⚠️ Failed to save log config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def enable_all(self):
        """Enable all log types."""
        for key in self.config:
            if key.startswith('show_'):
                self.config[key] = True
    
    def disable_all(self):
        """Disable all log types (except errors)."""
        for key in self.config:
            if key.startswith('show_'):
                self.config[key] = False
        self.config['show_error'] = True


# =============================================================================
# Main Logger Class
# =============================================================================

class Logger:
    """Main logger class."""
    
    # Singleton instance
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = LogConfig()
            self.log_file = None
            self.start_times = {}  # For timing measurements
            self._initialized = True
            self.open_log_file()
    
    def open_log_file(self):
        """Open log file for writing."""
        if self.config.get('log_to_file') and self.config.get('log_file'):
            try:
                log_file_path = self.config.get('log_file')
                if not os.path.isabs(log_file_path):
                    project_root = Path(__file__).parent
                    log_file_path = project_root / log_file_path
                
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                self.log_file = open(log_file_path, 
                                    self.config.get('file_mode', 'a'),
                                    encoding=self.config.get('file_encoding', 'utf-8'))
                
                # Write header
                timestamp = datetime.now().strftime(self.config.get('timestamp_format'))
                self.log_file.write(f"\n{'='*80}\n")
                self.log_file.write(f"LOG STARTED AT: {timestamp}\n")
                self.log_file.write(f"{'='*80}\n\n")
                
                self._log_file_opened(log_file_path)
                
            except Exception as e:
                self._log_file_open_error(e)
                self.log_file = None
    
    def _log_file_opened(self, filepath: str):
        """Internal method to log file opening."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.INFO}📝 Log file opened: {filepath}{LogColors.RESET}")
        else:
            print(f"📝 Log file opened: {filepath}")
    
    def _log_file_open_error(self, e: Exception):
        """Internal method to log file open error."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.WARNING}⚠️ Failed to open log file: {e}{LogColors.RESET}")
        else:
            print(f"⚠️ Failed to open log file: {e}")
    
    def close_log_file(self):
        """Close log file."""
        if self.log_file:
            try:
                timestamp = datetime.now().strftime(self.config.get('timestamp_format'))
                self.log_file.write(f"\n{'='*80}\n")
                self.log_file.write(f"LOG ENDED AT: {timestamp}\n")
                self.log_file.write(f"{'='*80}\n")
                self.log_file.close()
                self.log_file = None
                self._log_file_closed()
            except Exception as e:
                self._log_file_close_error(e)
    
    def _log_file_closed(self):
        """Internal method to log file closing."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.INFO}📝 Log file closed{LogColors.RESET}")
        else:
            print("📝 Log file closed")
    
    def _log_file_close_error(self, e: Exception):
        """Internal method to log file close error."""
        if self.config.get('use_colors', True):
            print(f"{LogColors.WARNING}⚠️ Failed to close log file: {e}{LogColors.RESET}")
        else:
            print(f"⚠️ Failed to close log file: {e}")
    
    def format_caller(self, tag: str, frame) -> str:
        """Format caller information."""
        max_length = self.config.get('caller_max_length', 20)
        
        # Truncate tag if too long
        if len(tag) > max_length:
            tag = tag[:max_length-3] + "..."
        elif len(tag) < max_length:
            tag = tag.ljust(max_length)
        
        # Add function name if available
        if frame:
            func_name = frame.f_code.co_name
            if func_name != '<module>':
                tag = f"{tag}.{func_name}"
                if len(tag) > max_length:
                    tag = tag[:max_length-3] + "..."
        
        return tag
    
    def get_color(self, log_type: str) -> str:
        """Get color code for log type."""
        if not self.config.get('use_colors', True):
            return ''
        
        color_map = {
            'ERROR': LogColors.ERROR,
            'WARNING': LogColors.WARNING,
            'INFO': LogColors.INFO,
            'SUCCESS': LogColors.SUCCESS,
            'DEBUG': LogColors.DEBUG,
            'DATA': LogColors.DATA,
            'TIMING': LogColors.TIMING,
        }
        
        return color_map.get(log_type, LogColors.RESET)
    
    def should_show(self, log_type: str) -> bool:
        """Check if log type should be shown."""
        config_key = f"show_{log_type.lower()}"
        return self.config.get(config_key, True)
    
    def format_message(self, message: Any, max_length: int = None) -> str:
        """Format message for display."""
        if max_length is None:
            max_length = self.config.get('max_message_length', 2000)
        
        # Convert to string
        if isinstance(message, (dict, list)):
            try:
                message_str = json.dumps(message, indent=self.config.get('indent_size', 2))
            except:
                message_str = str(message)
        else:
            message_str = str(message)
        
        # Handle multiline messages - ensure the first line stays with the log prefix
        # Don't add newline at the beginning for multiline data
        message_str = message_str.lstrip('\n')
        
        # Truncate if too long
        if len(message_str) > max_length:
            message_str = message_str[:max_length-3] + "..."
        
        return message_str
    
    def write_to_file(self, formatted_message: str):
        """Write log entry to file."""
        if self.log_file:
            try:
                # Remove color codes for file output
                import re
                clean_message = re.sub(r'\033\[[0-9;]*m', '', formatted_message)
                self.log_file.write(clean_message + '\n')
                self.log_file.flush()
            except Exception as e:
                # Don't print error to avoid recursion
                pass
    
    def log(self, tag: str, log_type: str, message: Any):
        """
        Main logging function.
        
        Args:
            tag: Caller identifier (e.g., "PPOController", "Main")
            log_type: Log type ("INFO", "WARNING", "ERROR", "SUCCESS", "DEBUG", "DATA", "TIMING")
            message: The message to log (can be any type)
        """
        # Check if this log type should be shown
        if not self.should_show(log_type):
            return
        
        # Get caller frame (skip internal frames)
        frame = None
        try:
            # Walk up the stack to find the first non-logging frame
            for f in inspect.stack()[1:]:
                filename = f.filename
                if 'logging_utils' not in filename and 'log' not in f.function:
                    frame = f.frame
                    break
        except:
            frame = None
        
        # Format caller tag
        caller = self.format_caller(tag, frame)
        
        # Get timestamp
        timestamp = datetime.now().strftime(self.config.get('timestamp_format'))
        
        # Format message
        message_str = self.format_message(message)
        
        # Get color
        color = self.get_color(log_type)
        reset = LogColors.RESET if self.config.get('use_colors', True) else ''
        
        # Left align log type with exactly 8 characters
        log_type_display = f"{log_type:<8}"
        
        # Format the log entry
        if self.config.get('use_colors', True):
            log_entry = f"{LogColors.DIM}{timestamp}{reset} " \
                       f"{color}[{log_type_display}]{reset} " \
                       f"{LogColors.BRIGHT_BLUE}{caller}{reset}: " \
                       f"{message_str}"
        else:
            log_entry = f"{timestamp} [{log_type_display}] {caller}: {message_str}"
        
        # Print to console
        print(log_entry)
        
        # Write to file
        self.write_to_file(log_entry)
    
    def start_timer(self, name: str):
        """Start a timer with given name."""
        self.start_times[name] = time.time()
        if self.should_show('timing'):
            self.log("Timer", "TIMING", f"Started timer '{name}'")
    
    def stop_timer(self, name: str) -> float:
        """Stop timer and return elapsed time in seconds."""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            del self.start_times[name]
            
            if self.should_show('timing'):
                self.log("Timer", "TIMING", 
                        f"Timer '{name}' completed in {elapsed:.4f} seconds")
            
            return elapsed
        else:
            self.log("Timer", "WARNING", f"Timer '{name}' not found")
            return 0.0
    
    def log_exception(self, tag: str, exception: Exception, context: str = ""):
        """Log an exception with traceback."""
        exc_type = type(exception).__name__
        exc_msg = str(exception)
        
        if context:
            message = f"{context} - {exc_type}: {exc_msg}"
        else:
            message = f"{exc_type}: {exc_msg}"
        
        self.log(tag, "ERROR", message)
        
        # Log full traceback in debug mode
        if self.should_show('debug'):
            traceback_str = traceback.format_exc()
            # Ensure traceback starts on the same line
            self.log(tag, "DEBUG", f"Traceback:\n{traceback_str}")
    
    def log_function_call(self, tag: str, func_name: str, args: tuple, kwargs: dict):
        """Log a function call with arguments."""
        if not self.should_show('debug'):
            return
        
        # Format arguments
        args_str = ", ".join([repr(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        
        all_args = []
        if args_str:
            all_args.append(args_str)
        if kwargs_str:
            all_args.append(kwargs_str)
        
        call_str = f"{func_name}({', '.join(all_args)})"
        self.log(tag, "DEBUG", f"Calling: {call_str}")
    
    def log_function_return(self, tag: str, func_name: str, return_value: Any):
        """Log a function return value."""
        if not self.should_show('debug'):
            return
        
        return_str = self.format_message(return_value, max_length=500)
        # Ensure return value stays on the same line even if multiline
        if '\n' in return_str:
            self.log(tag, "DEBUG", f"{func_name}() returned:\n{return_str}")
        else:
            self.log(tag, "DEBUG", f"{func_name}() returned: {return_str}")
    
    def log_data(self, tag: str, data_name: str, data: Any):
        """Log data with formatted display."""
        if not self.should_show('data'):
            return
        
        data_str = self.format_message(data)
        # Ensure multiline data starts on the same line
        if '\n' in data_str:
            self.log(tag, "DATA", f"{data_name} = {data_str}")
        else:
            self.log(tag, "DATA", f"{data_name} = {data_str}")
    
    def log_section(self, tag: str, title: str, width: int = 70):
        """Log a section header."""
        border = "=" * width
        
        # Log border, title, and border all in separate calls
        self.log(tag, "INFO", border)
        self.log(tag, "INFO", f"  {title.upper()}")
        self.log(tag, "INFO", border)


# =============================================================================
# Global Log Function
# =============================================================================

# Create global logger instance
_LOGGER = Logger()

def log(tag: str, log_type: str, message: Any):
    """
    Global log function.
    
    Args:
        tag: Caller identifier
        log_type: Log type
        message: Message to log
    """
    _LOGGER.log(tag, log_type, message)

def setup_logging(config_path: Optional[str] = None) -> Logger:
    """
    Set up logging with custom configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Logger instance
    """
    global _LOGGER
    _LOGGER = Logger()
    if config_path:
        _LOGGER.config.config_path = config_path
        _LOGGER.config.load_config()
    return _LOGGER

def get_logger() -> Logger:
    """Get the global logger instance."""
    return _LOGGER

# Convenience functions
def log_info(tag: str, message: Any):
    """Log info message."""
    log(tag, "INFO", message)

def log_warning(tag: str, message: Any):
    """Log warning message."""
    log(tag, "WARNING", message)

def log_error(tag: str, message: Any):
    """Log error message."""
    log(tag, "ERROR", message)

def log_success(tag: str, message: Any):
    """Log success message."""
    log(tag, "SUCCESS", message)

def log_debug(tag: str, message: Any):
    """Log debug message."""
    log(tag, "DEBUG", message)

def log_data(tag: str, data_name: str, data: Any):
    """Log data."""
    _LOGGER.log_data(tag, data_name, data)

def log_exception(tag: str, exception: Exception, context: str = ""):
    """Log exception."""
    _LOGGER.log_exception(tag, exception, context)

def log_function_call(tag: str, func_name: str, args: tuple, kwargs: dict):
    """Log function call."""
    _LOGGER.log_function_call(tag, func_name, args, kwargs)

def log_function_return(tag: str, func_name: str, return_value: Any):
    """Log function return."""
    _LOGGER.log_function_return(tag, func_name, return_value)

def start_timer(name: str):
    """Start timer."""
    _LOGGER.start_timer(name)

def stop_timer(name: str) -> float:
    """Stop timer and return elapsed time."""
    return _LOGGER.stop_timer(name)

def log_section(tag: str, title: str, width: int = 70):
    """Log section header."""
    _LOGGER.log_section(tag, title, width)

def close_logging():
    """Close logging resources."""
    _LOGGER.close_log_file()


# =============================================================================
# Context Manager for Function Logging
# =============================================================================

class LogFunction:
    """Context manager for logging function execution."""
    
    def __init__(self, tag: str, func_name: str, args: tuple = (), kwargs: dict = None):
        """
        Initialize function logger.
        
        Args:
            tag: Caller tag
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
        """
        self.tag = tag
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs or {}
        self.start_time = None
    
    def __enter__(self):
        """Enter context - log function call."""
        log_function_call(self.tag, self.func_name, self.args, self.kwargs)
        self.start_time = time.time()
        start_timer(f"{self.func_name}_execution")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log function completion."""
        elapsed = stop_timer(f"{self.func_name}_execution")
        
        if exc_type is None:
            log_success(self.tag, f"{self.func_name}() completed in {elapsed:.4f}s")
        else:
            log_error(self.tag, f"{self.func_name}() failed after {elapsed:.4f}s")
            if exc_val:
                log_exception(self.tag, exc_val, f"In {self.func_name}")
        
        # Don't suppress exceptions
        return False


# Decorator for automatic function logging
def logged_function(tag: str = None):
    """
    Decorator to automatically log function calls and returns.
    
    Args:
        tag: Optional tag (defaults to function module)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Determine tag
            func_tag = tag or func.__module__.split('.')[-1]
            
            # Log function call
            log_function_call(func_tag, func.__name__, args, kwargs)
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                elapsed = time.time() - start_time
                log_success(func_tag, 
                           f"{func.__name__}() completed in {elapsed:.4f}s")
                
                # Log return value
                log_function_return(func_tag, func.__name__, result)
                
                return result
                
            except Exception as e:
                # Log failure
                elapsed = time.time() - start_time
                log_error(func_tag, f"{func.__name__}() failed after {elapsed:.4f}s")
                log_exception(func_tag, e, f"In {func.__name__}")
                raise
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator


# =============================================================================
# Initialization
# =============================================================================

# Create log_config.json if it doesn't exist
def create_default_config():
    """Create default log config file."""
    config_path = Path(__file__).parent / "log_config.json"
    
    if not config_path.exists():
        default_config = LogConfig.DEFAULT_CONFIG
        default_config["log_file"] = "logs/training.log"
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            log_info("Logging", f"Created default config at {config_path}")
        except Exception as e:
            print(f"⚠️ Failed to create log config: {e}")

# Initialize on import
create_default_config()

# Export main functions
__all__ = [
    'log', 'log_info', 'log_warning', 'log_error', 'log_success',
    'log_debug', 'log_data', 'log_exception', 'log_function_call',
    'log_function_return', 'start_timer', 'stop_timer', 'log_section',
    'setup_logging', 'get_logger', 'close_logging', 'logged_function',
    'LogFunction', 'LogColors'
]
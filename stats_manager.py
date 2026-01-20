"""
Multi-Stage Statistics Manager with HDF5 Support.

Provides comprehensive logging for multi-agent, multi-stage evolutionary training
using PyTables for efficient HDF5 storage.

Multi-Process Safe with proper file locking and retry logic.
"""

import os
import sys
import json
import time
import fcntl
import numpy as np
import tables
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import logging
from logging_utils import (
    log_info, log_warning, log_error, log_success, 
    log_debug, log_data, log_exception, log_section,
    LogFunction
)


# =============================================================================
# MULTI-PROCESS FILE LOCKING UTILITIES
# =============================================================================

class FileLock:
    """
    Multi-process file lock using fcntl.flock.
    
    Provides safe concurrent access to files across multiple processes.
    Supports retry logic with exponential backoff.
    """
    
    def __init__(
        self,
        filepath: str,
        timeout: float = 30.0,
        max_retries: int = 10,
        retry_delay_base: float = 0.1,
        retry_delay_max: float = 2.0
    ):
        """
        Initialize file lock.
        
        Args:
            filepath: Path to file to lock
            timeout: Maximum time to wait for lock (seconds)
            max_retries: Maximum number of retry attempts
            retry_delay_base: Base delay for exponential backoff (seconds)
            retry_delay_max: Maximum delay between retries (seconds)
        """
        self.filepath = filepath
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.retry_delay_max = retry_delay_max
        
        # Internal file handle for locking
        self._lock_file = None
        self._locked = False
        
        log_debug("FileLock", f"Initialized for: {filepath}")
    
    def _get_lock_file_path(self) -> str:
        """Get path to lock file (adds .lock suffix)."""
        return f"{self.filepath}.lock"
    
    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the file lock.
        
        Args:
            blocking: If True, wait until lock is acquired. If False, return immediately.
            
        Returns:
            True if lock acquired, False if not (non-blocking mode only)
        """
        lock_path = self._get_lock_file_path()
        
        try:
            # Create lock directory if needed
            os.makedirs(os.path.dirname(lock_path) if os.path.dirname(lock_path) else '.', exist_ok=True)
            
            # Open lock file (create if doesn't exist)
            self._lock_file = open(lock_path, 'a')
            
            # Calculate retry parameters
            start_time = time.time()
            attempt = 0
            
            while True:
                try:
                    # Try to acquire exclusive lock (non-blocking by default)
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._locked = True
                    log_debug("FileLock", f"Acquired lock for: {self.filepath}")
                    return True
                    
                except (IOError, OSError) as e:
                    if e.errno in (11, 35):  # EWOULDBLOCK / EAGAIN
                        if not blocking:
                            log_debug("FileLock", f"Could not acquire lock (non-blocking): {self.filepath}")
                            return False
                        
                        # Check timeout
                        elapsed = time.time() - start_time
                        if elapsed >= self.timeout:
                            log_error("FileLock", f"Timeout waiting for lock: {self.filepath}")
                            raise TimeoutError(f"Could not acquire lock within {self.timeout}s: {self.filepath}")
                        
                        # Calculate delay with exponential backoff
                        attempt += 1
                        if attempt > self.max_retries:
                            attempt = self.max_retries
                        
                        delay = min(self.retry_delay_base * (2 ** (attempt - 1)), self.retry_delay_max)
                        delay = min(delay, self.timeout - elapsed)  # Don't exceed timeout
                        
                        log_debug("FileLock", f"Lock busy, waiting {delay:.3f}s (attempt {attempt}/{self.max_retries}): {self.filepath}")
                        time.sleep(delay)
                        
                    else:
                        raise
                        
        except Exception as e:
            log_exception("FileLock", e, f"Error acquiring lock: {self.filepath}")
            self._cleanup()
            raise
    
    def release(self) -> bool:
        """Release the file lock."""
        try:
            if self._locked and self._lock_file:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                log_debug("FileLock", f"Released lock for: {self.filepath}")
                self._locked = False
                
            self._cleanup()
            return True
            
        except Exception as e:
            log_exception("FileLock", e, f"Error releasing lock: {self.filepath}")
            self._cleanup()
            return False
    
    def _cleanup(self):
        """Clean up lock file handle."""
        if self._lock_file:
            try:
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None
        self._locked = False
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
    
    def __del__(self):
        """Destructor."""
        self.release()


def with_file_lock(
    filepath: str,
    timeout: float = 30.0,
    max_retries: int = 10
) -> Callable:
    """
    Decorator to add file locking to a function.
    
    Args:
        filepath: Path to file to lock
        timeout: Maximum time to wait for lock (seconds)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            lock = FileLock(filepath, timeout=timeout, max_retries=max_retries)
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def safe_file_operation(
    filepath: str,
    timeout: float = 30.0,
    max_retries: int = 10
):
    """
    Context manager for safe file operations with locking.
    
    Args:
        filepath: Path to file to lock
        timeout: Maximum time to wait for lock (seconds)
        max_retries: Maximum number of retry attempts
    """
    lock = FileLock(filepath, timeout=timeout, max_retries=max_retries)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


# =============================================================================
# ATOMIC FILE SAVE UTILITIES
# =============================================================================

def atomic_save(
    data: Any,
    filepath: str,
    save_func: Callable = None,
    timeout: float = 30.0,
    max_retries: int = 10
) -> bool:
    """
    Atomically save data to a file.
    
    Writes to a temporary file first, then renames to the target file.
    This ensures atomicity - either the entire write succeeds or it doesn't.
    
    Args:
        data: Data to save
        filepath: Target file path
        save_func: Function to save data (e.g., torch.save, pickle.dump)
        timeout: Maximum time to wait for lock (seconds)
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if save succeeded, False otherwise
    """
    with LogFunction("FileUtils", "atomic_save", args=(filepath,)):
        
        log_info("FileUtils", f"Atomically saving to: {filepath}")
        
        # Create temp file in same directory (for atomic rename)
        temp_path = f"{filepath}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
        
        lock = FileLock(filepath, timeout=timeout, max_retries=max_retries)
        
        try:
            # Acquire lock
            lock.acquire()
            
            try:
                # Save to temp file
                if save_func:
                    save_func(data, temp_path)
                else:
                    # Default to torch.save if available
                    import torch
                    torch.save(data, temp_path)
                
                # Atomic rename
                os.rename(temp_path, filepath)
                
                log_success("FileUtils", f"Atomically saved: {filepath}")
                return True
                
            except Exception as e:
                log_exception("FileUtils", e, f"Error during save to {filepath}")
                
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                
                return False
                
        except Exception as e:
            log_exception("FileUtils", e, f"Could not acquire lock for {filepath}")
            return False
            
        finally:
            lock.release()


def safe_torch_save(
    model: Any,
    filepath: str,
    timeout: float = 30.0,
    max_retries: int = 10
) -> bool:
    """
    Safely save PyTorch model with file locking.
    
    Args:
        model: PyTorch model or checkpoint dict
        filepath: Target file path
        timeout: Maximum time to wait for lock (seconds)
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if save succeeded, False otherwise
    """
    return atomic_save(
        data=model,
        filepath=filepath,
        save_func=lambda m, p: __import__('torch').save(m, p),
        timeout=timeout,
        max_retries=max_retries
    )


def safe_torch_load(
    filepath: str,
    map_location: Any = None,
    timeout: float = 30.0,
    max_retries: int = 10
) -> Any:
    """
    Safely load PyTorch model with file locking and retries.
    
    Args:
        filepath: Path to checkpoint file
        map_location: Optional device mapping
        timeout: Maximum time to wait for lock (seconds)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Loaded model/checkpoint
    """
    with LogFunction("FileUtils", "safe_torch_load", args=(filepath,)):
        
        log_info("FileUtils", f"Loading checkpoint: {filepath}")
        
        lock = FileLock(filepath, timeout=timeout, max_retries=max_retries)
        
        # Try to acquire lock (non-blocking first, then with retries)
        if not lock.acquire(blocking=False):
            # Lock busy, wait with retries
            log_warning("FileUtils", f"File locked, waiting to load: {filepath}")
            lock.acquire(blocking=True)
        
        try:
            import torch
            
            # Try to load with retries
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    if map_location:
                        checkpoint = torch.load(filepath, map_location=map_location)
                    else:
                        checkpoint = torch.load(filepath)
                    
                    log_success("FileUtils", f"Loaded checkpoint: {filepath}")
                    return checkpoint
                    
                except (EOFError, IOError) as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = 0.1 * (2 ** attempt)
                        log_warning("FileUtils", f"Load failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.2f}s: {filepath}")
                        time.sleep(delay)
                    else:
                        raise
            
            raise last_error
            
        except Exception as e:
            log_exception("FileUtils", e, f"Error loading checkpoint: {filepath}")
            raise
            
        finally:
            lock.release()


# HDF5 Description Definitions
STAGES_TABLE_DESC = {
    'stage_id': tables.Int32Col(pos=0),
    'stage_name': tables.StringCol(64, pos=1),
    'start_episode_global': tables.Int64Col(pos=2),
    'end_episode_global': tables.Int64Col(pos=3),
    'timestamp': tables.StringCol(32, pos=4),
    'status': tables.StringCol(32, pos=5),
    'hyperparameters': tables.StringCol(4096, pos=6),  # JSON string
    'metrics': tables.StringCol(4096, pos=7),  # JSON string
}

EPISODES_TABLE_DESC = {
    'global_episode_id': tables.Int64Col(pos=0),
    'stage_id': tables.Int32Col(pos=1),
    'local_episode_id': tables.Int32Col(pos=2),
    'total_steps': tables.Int32Col(pos=3),
    'total_reward': tables.Float32Col(pos=4),
    'success': tables.BoolCol(pos=5),
    'termination_reason': tables.StringCol(64, pos=6),
    'agent_id': tables.StringCol(64, pos=7),
    'start_idx': tables.Int64Col(pos=8),
    'end_idx': tables.Int64Col(pos=9),
}

TIMESTEPS_TABLE_DESC = {
    'global_episode_id': tables.Int64Col(pos=0),
    'stage_id': tables.Int32Col(pos=1),
    'local_timestep': tables.Int32Col(pos=2),
    'global_timestep': tables.Int64Col(pos=3),
    'reward': tables.Float32Col(pos=4),
    'action': tables.Float32Col(shape=10, pos=5),  # Max 10 joints
    'agent_id': tables.StringCol(64, pos=6),
    'done': tables.BoolCol(pos=7),
}

AGENTS_TABLE_DESC = {
    'agent_id': tables.StringCol(64, pos=0),
    'stage_id': tables.Int32Col(pos=1),
    'agent_type': tables.StringCol(32, pos=2),
    'parameters_hash': tables.StringCol(128, pos=3),
    'parent_id': tables.StringCol(64, pos=4),
    'creation_timestamp': tables.StringCol(32, pos=5),
    'lineage_depth': tables.Int32Col(pos=6),
}

STAGE_METRICS_TABLE_DESC = {
    'stage_id': tables.Int32Col(pos=0),
    'metric_name': tables.StringCol(64, pos=1),
    'window_size': tables.Int32Col(pos=2),
    'value': tables.Float32Col(pos=3),
    'timestamp': tables.StringCol(32, pos=4),
}


@dataclass
class StageInfo:
    """Information about a training stage."""
    stage_id: int
    stage_name: str
    start_episode_global: int
    end_episode_global: Optional[int] = None
    timestamp: str = ""
    status: str = "running"
    hyperparameters: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)


@dataclass
class EpisodeInfo:
    """Information about a training episode."""
    global_episode_id: int
    stage_id: int
    local_episode_id: int
    total_steps: int
    total_reward: float
    success: bool
    termination_reason: str
    agent_id: str = ""
    start_idx: int = 0
    end_idx: int = 0


@dataclass
class AgentInfo:
    """Information about an agent."""
    agent_id: str
    stage_id: int
    agent_type: str  # 'ddpg' or 'ppo'
    parameters_hash: str = ""
    parent_id: str = ""
    creation_timestamp: str = ""
    lineage_depth: int = 0


class HDF5StatsLogger:
    """
    HDF5-based statistics logger for multi-stage training.

    Provides efficient storage and querying of:
    - Stage metadata
    - Episode results with stage context
    - Per-timestep data with agent tracking
    - Agent lineage information
    - Aggregated stage metrics

    Multi-process safe with proper file locking.
    """

    def __init__(
        self,
        filepath: str,
        mode: str = 'w',
        complevel: int = 9,
        complib: str = 'blosc',
        lock_timeout: float = 60.0,
        lock_max_retries: int = 20
    ):
        """
        Initialize HDF5 statistics logger.

        Args:
            filepath: Path to HDF5 file
            mode: 'w' (write), 'a' (append), 'r' (read)
            complevel: Compression level (0-9)
            complib: Compression library
            lock_timeout: Maximum time to wait for file lock (seconds)
            lock_max_retries: Maximum number of lock retry attempts
        """
        log_section("HDF5StatsLogger", "INITIALIZING STATS LOGGER")

        with LogFunction("HDF5StatsLogger", "__init__",
                        args=(filepath, mode, complevel, complib)):

            self.filepath = filepath
            self.mode = mode
            self._lock_timeout = lock_timeout
            self._lock_max_retries = lock_max_retries

            # Initialize file lock for multi-process safety
            self._file_lock = FileLock(
                filepath,
                timeout=lock_timeout,
                max_retries=lock_max_retries
            )

            log_info("HDF5StatsLogger", f"HDF5 file: {filepath}")
            log_info("HDF5StatsLogger", f"Mode: {mode}")
            log_info("HDF5StatsLogger", f"Compression: {complib} level {complevel}")
            log_info("HDF5StatsLogger", f"Lock timeout: {lock_timeout}s, max retries: {lock_max_retries}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Acquire file lock before opening HDF5 file
            acquired_lock = self._file_lock.acquire(blocking=True)
            if not acquired_lock:
                raise RuntimeError(f"Could not acquire file lock for {filepath} within {lock_timeout}s")

            try:
                # Open/create HDF5 file
                self.h5file = tables.open_file(
                    filepath,
                    mode=mode,
                    title="Multi-Stage Training Statistics",
                    filters=tables.Filters(
                        complevel=complevel,
                        complib=complib
                    )
                )

                # Initialize tables
                self._init_tables()

                # Track counters
                self._global_episode_counter = 0
                self._global_timestep_counter = 0
                self._current_stage_id = -1

                # Load existing counters if appending
                if mode == 'a':
                    self._load_counters()

                log_data("HDF5StatsLogger", "Initial counters", {
                    "global_episode_counter": self._global_episode_counter,
                    "global_timestep_counter": self._global_timestep_counter,
                    "current_stage_id": self._current_stage_id
                })

                log_success("HDF5StatsLogger", "Stats logger initialized")

            except Exception as e:
                self._file_lock.release()
                raise
    
    def _init_tables(self):
        """Initialize all required HDF5 tables."""
        log_debug("HDF5StatsLogger", "Initializing HDF5 tables")
        
        try:
            # Stages table
            if 'stages' not in self.h5file.root:
                self.stages_table = self.h5file.create_table(
                    self.h5file.root,
                    'stages',
                    STAGES_TABLE_DESC,
                    "Training stages"
                )
                log_debug("HDF5StatsLogger", "Created stages table")
            else:
                self.stages_table = self.h5file.root.stages
                log_debug("HDF5StatsLogger", "Loaded existing stages table")
            
            # Episodes table
            if 'episodes' not in self.h5file.root:
                self.episodes_table = self.h5file.create_table(
                    self.h5file.root,
                    'episodes',
                    EPISODES_TABLE_DESC,
                    "Episode results"
                )
                log_debug("HDF5StatsLogger", "Created episodes table")
            else:
                self.episodes_table = self.h5file.root.episodes
                log_debug("HDF5StatsLogger", "Loaded existing episodes table")
            
            # Timesteps table
            if 'timesteps' not in self.h5file.root:
                self.timesteps_table = self.h5file.create_table(
                    self.h5file.root,
                    'timesteps',
                    TIMESTEPS_TABLE_DESC,
                    "Per-timestep data"
                )
                log_debug("HDF5StatsLogger", "Created timesteps table")
            else:
                self.timesteps_table = self.h5file.root.timesteps
                log_debug("HDF5StatsLogger", "Loaded existing timesteps table")
            
            # Agents table
            if 'agents' not in self.h5file.root:
                self.agents_table = self.h5file.create_table(
                    self.h5file.root,
                    'agents',
                    AGENTS_TABLE_DESC,
                    "Agent information"
                )
                log_debug("HDF5StatsLogger", "Created agents table")
            else:
                self.agents_table = self.h5file.root.agents
                log_debug("HDF5StatsLogger", "Loaded existing agents table")
            
            # Stage metrics table
            if 'stage_metrics' not in self.h5file.root:
                self.stage_metrics_table = self.h5file.create_table(
                    self.h5file.root,
                    'stage_metrics',
                    STAGE_METRICS_TABLE_DESC,
                    "Aggregated stage metrics"
                )
                log_debug("HDF5StatsLogger", "Created stage_metrics table")
            else:
                self.stage_metrics_table = self.h5file.root.stage_metrics
                log_debug("HDF5StatsLogger", "Loaded existing stage_metrics table")
            
            log_success("HDF5StatsLogger", "All tables initialized")
            
        except Exception as e:
            log_exception("HDF5StatsLogger", e, "Failed to initialize tables")
            raise
    
    def _load_counters(self):
        """Load existing counters from data."""
        log_debug("HDF5StatsLogger", "Loading existing counters")
        
        try:
            # Get max episode ID
            if len(self.episodes_table) > 0:
                self._global_episode_counter = self.episodes_table.cols.global_episode_id[-1] + 1
                log_debug("HDF5StatsLogger", f"Loaded global_episode_counter: {self._global_episode_counter}")
            
            # Get max timestep ID
            if len(self.timesteps_table) > 0:
                self._global_timestep_counter = self.timesteps_table.cols.global_timestep[-1] + 1
                log_debug("HDF5StatsLogger", f"Loaded global_timestep_counter: {self._global_timestep_counter}")
            
            # Get current stage
            if len(self.stages_table) > 0:
                self._current_stage_id = self.stages_table.cols.stage_id[-1]
                log_debug("HDF5StatsLogger", f"Loaded current_stage_id: {self._current_stage_id}")
                
        except Exception as e:
            log_exception("HDF5StatsLogger", e, "Failed to load counters")
            # Reset counters on error
            self._global_episode_counter = 0
            self._global_timestep_counter = 0
            self._current_stage_id = -1
    
    # ==================== Stage Methods ====================
    
    def log_stage(
        self,
        stage_info: StageInfo
    ) -> int:
        """
        Log a new training stage.
        
        Args:
            stage_info: StageInfo dataclass
            
        Returns:
            Row index in table
        """
        with LogFunction("HDF5StatsLogger", "log_stage",
                        args=(),
                        kwargs={'stage_id': stage_info.stage_id,
                               'stage_name': stage_info.stage_name}):
            
            log_info("HDF5StatsLogger", f"Logging stage: {stage_info.stage_name} (ID: {stage_info.stage_id})")
            
            row = self.stages_table.row
            
            row['stage_id'] = stage_info.stage_id
            row['stage_name'] = stage_info.stage_name
            row['start_episode_global'] = stage_info.start_episode_global
            row['end_episode_global'] = stage_info.end_episode_global or -1
            row['timestamp'] = stage_info.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row['status'] = stage_info.status
            row['hyperparameters'] = json.dumps(stage_info.hyperparameters)
            row['metrics'] = json.dumps(stage_info.metrics)
            
            row.append()
            self.stages_table.flush()
            
            self._current_stage_id = stage_info.stage_id
            
            row_index = len(self.stages_table) - 1
            
            log_success("HDF5StatsLogger", f"Logged stage {stage_info.stage_id} at row {row_index}")
            log_data("HDF5StatsLogger", "Stage info", {
                'stage_id': stage_info.stage_id,
                'name': stage_info.stage_name,
                'start_episode': stage_info.start_episode_global,
                'status': stage_info.status
            })
            
            return row_index
    
    def update_stage(
        self,
        stage_id: int,
        end_episode_global: Optional[int] = None,
        status: Optional[str] = None,
        metrics: Optional[Dict] = None
    ):
        """Update stage information (e.g., on completion)."""
        with LogFunction("HDF5StatsLogger", "update_stage",
                        args=(stage_id, end_episode_global, status),
                        kwargs={'has_metrics': metrics is not None}):
            
            log_info("HDF5StatsLogger", f"Updating stage: {stage_id}")
            
            if not self.stages_table.row:
                log_warning("HDF5StatsLogger", "Stages table has no rows")
                return
            
            # Read all stages
            stages_data = []
            for i in range(len(self.stages_table)):
                row = self.stages_table[i]
                stages_data.append({
                    'stage_id': row['stage_id'],
                    'stage_name': row['stage_name'],
                    'start_episode_global': row['start_episode_global'],
                    'end_episode_global': row['end_episode_global'],
                    'timestamp': row['timestamp'],
                    'status': row['status'],
                    'hyperparameters': row['hyperparameters'],
                    'metrics': row['metrics']
                })
            
            # Find and update the matching stage
            stage_found = False
            for i, data in enumerate(stages_data):
                if data['stage_id'] == stage_id:
                    if end_episode_global is not None:
                        data['end_episode_global'] = end_episode_global
                    if status is not None:
                        data['status'] = status
                    if metrics is not None:
                        data['metrics'] = json.dumps(metrics)
                    stage_found = True
                    log_debug("HDF5StatsLogger", f"Found stage {stage_id} at index {i}")
                    break
            
            if not stage_found:
                log_warning("HDF5StatsLogger", f"Stage {stage_id} not found")
                return
            
            # Clear and re-write the table
            self.stages_table.remove_rows(0)
            self.stages_table.flush()
            
            for data in stages_data:
                row = self.stages_table.row
                row['stage_id'] = data['stage_id']
                row['stage_name'] = data['stage_name']
                row['start_episode_global'] = data['start_episode_global']
                row['end_episode_global'] = data['end_episode_global']
                row['timestamp'] = data['timestamp']
                row['status'] = data['status']
                row['hyperparameters'] = data['hyperparameters']
                row['metrics'] = data['metrics']
                row.append()
            
            self.stages_table.flush()
            
            log_success("HDF5StatsLogger", f"Updated stage {stage_id}")
            log_data("HDF5StatsLogger", "Update details", {
                'end_episode': end_episode_global,
                'status': status,
                'has_metrics': metrics is not None
            })
    
    # ==================== Episode Methods ====================
    
    def log_episode(
        self,
        episode_info: EpisodeInfo
    ) -> int:
        """
        Log a training episode.
        
        Args:
            episode_info: EpisodeInfo dataclass
            
        Returns:
            Global episode ID
        """
        with LogFunction("HDF5StatsLogger", "log_episode",
                        args=(),
                        kwargs={'stage_id': episode_info.stage_id,
                               'agent_id': episode_info.agent_id,
                               'total_reward': episode_info.total_reward}):
            
            log_info("HDF5StatsLogger", f"Logging episode: stage={episode_info.stage_id}, agent={episode_info.agent_id}")
            
            row = self.episodes_table.row
            
            # Use provided global ID or auto-increment
            global_ep_id = episode_info.global_episode_id
            if global_ep_id < 0:
                global_ep_id = self._global_episode_counter
                self._global_episode_counter += 1
            
            row['global_episode_id'] = global_ep_id
            row['stage_id'] = episode_info.stage_id
            row['local_episode_id'] = episode_info.local_episode_id
            row['total_steps'] = episode_info.total_steps
            row['total_reward'] = episode_info.total_reward
            row['success'] = episode_info.success
            row['termination_reason'] = episode_info.termination_reason
            row['agent_id'] = episode_info.agent_id
            row['start_idx'] = episode_info.start_idx
            row['end_idx'] = episode_info.end_idx
            
            row.append()
            self.episodes_table.flush()
            
            log_success("HDF5StatsLogger", f"Logged episode {global_ep_id}")
            log_data("HDF5StatsLogger", "Episode details", {
                'global_id': global_ep_id,
                'stage_id': episode_info.stage_id,
                'agent_id': episode_info.agent_id,
                'reward': episode_info.total_reward,
                'success': episode_info.success,
                'steps': episode_info.total_steps
            })
            
            return global_ep_id
    
    def log_timesteps(
        self,
        global_episode_id: int,
        stage_id: int,
        local_episode_id: int,
        rewards: List[float],
        actions: List[List[float]],
        agent_id: str,
        dones: Optional[List[bool]] = None
    ) -> Tuple[int, int]:
        """
        Log per-timestep data for an episode.
        
        Args:
            global_episode_id: Global episode identifier
            stage_id: Current stage ID
            local_episode_id: Local episode ID within stage
            rewards: List of rewards per timestep
            actions: List of actions per timestep
            agent_id: Agent identifier
            dones: List of done flags (default: all False except last)
            
        Returns:
            (start_timestep_idx, end_timestep_idx)
        """
        with LogFunction("HDF5StatsLogger", "log_timesteps",
                        args=(global_episode_id, stage_id, local_episode_id, agent_id),
                        kwargs={'num_rewards': len(rewards),
                               'num_actions': len(actions)}):
            
            log_info("HDF5StatsLogger", f"Logging timesteps: episode={global_episode_id}, agent={agent_id}")
            
            if dones is None:
                dones = [False] * len(rewards)
                if rewards:
                    dones[-1] = True
            
            start_idx = self._global_timestep_counter
            
            log_debug("HDF5StatsLogger", f"Starting at timestep index: {start_idx}")
            log_debug("HDF5StatsLogger", f"Logging {len(rewards)} timesteps")
            
            for t, (reward, action, done) in enumerate(zip(rewards, actions, dones)):
                row = self.timesteps_table.row
                
                row['global_episode_id'] = global_episode_id
                row['stage_id'] = stage_id
                row['local_timestep'] = t
                row['global_timestep'] = self._global_timestep_counter
                row['reward'] = reward
                
                # Pad/truncate action to max size
                action_padded = np.zeros(10, dtype=np.float32)
                action_padded[:len(action)] = action
                row['action'] = action_padded
                
                row['agent_id'] = agent_id
                row['done'] = done
                
                row.append()
                
                self._global_timestep_counter += 1
            
            self.timesteps_table.flush()
            
            end_idx = self._global_timestep_counter - 1
            
            log_success("HDF5StatsLogger", f"Logged {len(rewards)} timesteps")
            log_data("HDF5StatsLogger", "Timestep range", {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_timesteps': len(rewards)
            })
            
            return start_idx, end_idx
    
    # ==================== Agent Methods ====================
    
    def log_agent(
        self,
        agent_info: AgentInfo
    ):
        """Log agent information."""
        with LogFunction("HDF5StatsLogger", "log_agent",
                        args=(),
                        kwargs={'agent_id': agent_info.agent_id,
                               'stage_id': agent_info.stage_id}):
            
            log_info("HDF5StatsLogger", f"Logging agent: {agent_info.agent_id}")
            
            row = self.agents_table.row
            
            row['agent_id'] = agent_info.agent_id
            row['stage_id'] = agent_info.stage_id
            row['agent_type'] = agent_info.agent_type
            row['parameters_hash'] = agent_info.parameters_hash
            row['parent_id'] = agent_info.parent_id
            row['creation_timestamp'] = agent_info.creation_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row['lineage_depth'] = agent_info.lineage_depth
            
            row.append()
            self.agents_table.flush()
            
            log_success("HDF5StatsLogger", f"Logged agent: {agent_info.agent_id}")
            log_data("HDF5StatsLogger", "Agent info", {
                'agent_id': agent_info.agent_id,
                'stage_id': agent_info.stage_id,
                'agent_type': agent_info.agent_type,
                'parent_id': agent_info.parent_id,
                'lineage_depth': agent_info.lineage_depth
            })
    
    # ==================== Stage Metrics Methods ====================
    
    def log_stage_metric(
        self,
        stage_id: int,
        metric_name: str,
        value: float,
        window_size: int = 0,
        timestamp: Optional[str] = None
    ):
        """Log aggregated stage metric."""
        with LogFunction("HDF5StatsLogger", "log_stage_metric",
                        args=(stage_id, metric_name, value, window_size)):
            
            log_info("HDF5StatsLogger", f"Logging metric: stage={stage_id}, metric={metric_name}, value={value}")
            
            row = self.stage_metrics_table.row
            
            row['stage_id'] = stage_id
            row['metric_name'] = metric_name
            row['window_size'] = window_size
            row['value'] = value
            row['timestamp'] = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            row.append()
            self.stage_metrics_table.flush()
            
            log_debug("HDF5StatsLogger", f"Logged metric {metric_name}={value} for stage {stage_id}")
    
    def log_stage_summary(
        self,
        stage_id: int,
        mean_reward: float,
        success_rate: float,
        mean_episode_length: float,
        window_size: int = 100
    ):
        """Log complete stage summary."""
        with LogFunction("HDF5StatsLogger", "log_stage_summary",
                        args=(stage_id, mean_reward, success_rate, mean_episode_length, window_size)):
            
            log_info("HDF5StatsLogger", f"Logging stage summary: stage={stage_id}")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.log_stage_metric(stage_id, 'mean_reward', mean_reward, window_size, timestamp)
            self.log_stage_metric(stage_id, 'success_rate', success_rate, window_size, timestamp)
            self.log_stage_metric(stage_id, 'mean_episode_length', mean_episode_length, window_size, timestamp)
            
            log_success("HDF5StatsLogger", f"Logged stage {stage_id} summary")
            log_data("HDF5StatsLogger", "Stage summary", {
                'stage_id': stage_id,
                'mean_reward': mean_reward,
                'success_rate': success_rate,
                'mean_episode_length': mean_episode_length,
                'window_size': window_size
            })
    
    # ==================== Query Methods ====================
    
    def get_stage_stats(self, stage_id: int) -> Dict:
        """Get statistics for a specific stage."""
        with LogFunction("HDF5StatsLogger", "get_stage_stats", args=(stage_id,)):
            log_info("HDF5StatsLogger", f"Getting stats for stage: {stage_id}")
            
            # Get stage info
            stage_data = None
            for i in range(len(self.stages_table)):
                if self.stages_table[i]['stage_id'] == stage_id:
                    row = self.stages_table[i]
                    stage_data = {
                        'stage_id': row['stage_id'],
                        'stage_name': row['stage_name'],
                        'start_episode': row['start_episode_global'],
                        'end_episode': row['end_episode_global'],
                        'status': row['status'],
                        'hyperparameters': json.loads(row['hyperparameters']),
                        'metrics': json.loads(row['metrics'])
                    }
                    break
            
            if stage_data is None:
                log_warning("HDF5StatsLogger", f"Stage {stage_id} not found")
                return {}
            
            # Get episode stats for this stage
            episodes = self.episodes_table.where(f'stage_id == {stage_id}')
            
            rewards = []
            successes = []
            lengths = []
            
            for ep in episodes:
                rewards.append(ep['total_reward'])
                successes.append(ep['success'])
                lengths.append(ep['total_steps'])
            
            stage_data['num_episodes'] = len(rewards)
            stage_data['mean_reward'] = float(np.mean(rewards)) if rewards else 0.0
            stage_data['std_reward'] = float(np.std(rewards)) if rewards else 0.0
            stage_data['success_rate'] = float(np.mean(successes)) if successes else 0.0
            stage_data['mean_episode_length'] = float(np.mean(lengths)) if lengths else 0.0
            
            log_data("HDF5StatsLogger", f"Stage {stage_id} stats", {
                'num_episodes': stage_data['num_episodes'],
                'mean_reward': stage_data['mean_reward'],
                'success_rate': stage_data['success_rate']
            })
            
            return stage_data
    
    def get_agent_lineage(self, agent_id: str) -> List[Dict]:
        """Trace lineage of an agent back to ancestors."""
        with LogFunction("HDF5StatsLogger", "get_agent_lineage", args=(agent_id,)):
            log_info("HDF5StatsLogger", f"Getting lineage for agent: {agent_id}")
            
            lineage = []
            
            current_id = agent_id
            max_depth = 20  # Prevent infinite loops
            
            while current_id and len(lineage) < max_depth:
                # Find agent in table
                found = False
                for i in range(len(self.agents_table)):
                    row = self.agents_table[i]
                    if row['agent_id'] == current_id:
                        lineage.append({
                            'agent_id': row['agent_id'],
                            'stage_id': row['stage_id'],
                            'parent_id': row['parent_id'],
                            'lineage_depth': row['lineage_depth'],
                            'timestamp': row['creation_timestamp']
                        })
                        current_id = row['parent_id']
                        found = True
                        break
                
                if not found:
                    break
            
            log_info("HDF5StatsLogger", f"Found lineage of length {len(lineage)} for agent {agent_id}")
            log_data("HDF5StatsLogger", "Lineage", lineage)
            
            return lineage
    
    def get_all_agents_in_stage(self, stage_id: int) -> List[str]:
        """Get all unique agent IDs in a stage."""
        with LogFunction("HDF5StatsLogger", "get_all_agents_in_stage", args=(stage_id,)):
            log_info("HDF5StatsLogger", f"Getting agents in stage: {stage_id}")
            
            agents = set()
            
            for ep in self.episodes_table.where(f'stage_id == {stage_id}'):
                agents.add(ep['agent_id'])
            
            agent_list = list(agents)
            
            log_info("HDF5StatsLogger", f"Found {len(agent_list)} agents in stage {stage_id}")
            log_data("HDF5StatsLogger", "Agent IDs", agent_list)
            
            return agent_list
    
    def get_episodes_for_agent(
        self,
        agent_id: str,
        stage_id: Optional[int] = None
    ) -> List[Dict]:
        """Get all episodes for a specific agent."""
        with LogFunction("HDF5StatsLogger", "get_episodes_for_agent",
                        args=(agent_id,),
                        kwargs={'stage_id': stage_id}):
            
            log_info("HDF5StatsLogger", f"Getting episodes for agent: {agent_id}, stage: {stage_id}")
            
            episodes = []
            
            if stage_id is not None:
                condition = f'(stage_id == {stage_id}) & (agent_id == b"{agent_id}")'
            else:
                condition = f'agent_id == b"{agent_id}"'
            
            for ep in self.episodes_table.where(condition):
                episodes.append({
                    'global_episode_id': ep['global_episode_id'],
                    'stage_id': ep['stage_id'],
                    'local_episode_id': ep['local_episode_id'],
                    'total_steps': ep['total_steps'],
                    'total_reward': ep['total_reward'],
                    'success': ep['success'],
                    'termination_reason': ep['termination_reason']
                })
            
            log_info("HDF5StatsLogger", f"Found {len(episodes)} episodes for agent {agent_id}")
            log_data("HDF5StatsLogger", "Episode count", len(episodes))
            
            return episodes
    
    def get_best_agent_in_stage(
        self,
        stage_id: int,
        metric: str = 'total_reward'
    ) -> Tuple[Optional[str], float]:
        """Get best performing agent in a stage."""
        with LogFunction("HDF5StatsLogger", "get_best_agent_in_stage",
                        args=(stage_id, metric)):
            
            log_info("HDF5StatsLogger", f"Getting best agent in stage {stage_id} by {metric}")
            
            agent_scores = {}
            
            for ep in self.episodes_table.where(f'stage_id == {stage_id}'):
                agent_id = ep['agent_id']
                score = ep[metric]
                
                if agent_id not in agent_scores:
                    agent_scores[agent_id] = []
                agent_scores[agent_id].append(score)
            
            if not agent_scores:
                log_warning("HDF5StatsLogger", f"No episodes found for stage {stage_id}")
                return None, 0.0
            
            # Average score per agent
            agent_avgs = {k: np.mean(v) for k, v in agent_scores.items()}
            
            best_agent = max(agent_avgs, key=agent_avgs.get)
            best_score = agent_avgs[best_agent]
            
            log_success("HDF5StatsLogger", f"Best agent in stage {stage_id}: {best_agent} (score: {best_score})")
            log_data("HDF5StatsLogger", "Agent scores", {
                'best_agent': best_agent,
                'best_score': best_score,
                'num_agents': len(agent_avgs)
            })
            
            return best_agent, best_score
    
    def get_progress(self) -> Dict:
        """Get overall training progress."""
        with LogFunction("HDF5StatsLogger", "get_progress"):
            log_info("HDF5StatsLogger", "Getting training progress")
            
            progress = {
                'total_episodes': len(self.episodes_table),
                'total_timesteps': len(self.timesteps_table),
                'current_stage': self._current_stage_id,
                'stages_completed': 0,
                'stages_in_progress': 0
            }
            
            # Count stages by status
            for i in range(len(self.stages_table)):
                row = self.stages_table[i]
                if row['status'] == 'completed':
                    progress['stages_completed'] += 1
                else:
                    progress['stages_in_progress'] += 1
            
            log_data("HDF5StatsLogger", "Training progress", progress)
            
            return progress
    
    # ==================== Export Methods ====================
    
    def export_episodes_csv(self, filepath: str, stage_id: Optional[int] = None):
        """Export episodes to CSV format."""
        with LogFunction("HDF5StatsLogger", "export_episodes_csv",
                        args=(filepath,),
                        kwargs={'stage_id': stage_id}):
            
            log_info("HDF5StatsLogger", f"Exporting episodes to CSV: {filepath}")
            
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'global_episode_id', 'stage_id', 'local_episode_id',
                    'total_steps', 'total_reward', 'success',
                    'termination_reason', 'agent_id'
                ])
                
                if stage_id is not None:
                    episodes = self.episodes_table.where(f'stage_id == {stage_id}')
                else:
                    episodes = self.episodes_table.iterrows()
                
                episode_count = 0
                for ep in episodes:
                    writer.writerow([
                        ep['global_episode_id'],
                        ep['stage_id'],
                        ep['local_episode_id'],
                        ep['total_steps'],
                        ep['total_reward'],
                        ep['success'],
                        ep['termination_reason'],
                        ep['agent_id']
                    ])
                    episode_count += 1
                
                log_success("HDF5StatsLogger", f"Exported {episode_count} episodes to {filepath}")
    
    def get_dataframe(self, table_name: str):
        """Get table as pandas DataFrame (requires pandas)."""
        with LogFunction("HDF5StatsLogger", "get_dataframe", args=(table_name,)):
            log_info("HDF5StatsLogger", f"Getting DataFrame for table: {table_name}")
            
            try:
                import pandas as pd
            except ImportError:
                log_error("HDF5StatsLogger", "pandas is required for DataFrame export")
                raise ImportError("pandas is required for DataFrame export")
            
            if table_name == 'stages':
                table = self.stages_table
            elif table_name == 'episodes':
                table = self.episodes_table
            elif table_name == 'timesteps':
                table = self.timesteps_table
            elif table_name == 'agents':
                table = self.agents_table
            elif table_name == 'stage_metrics':
                table = self.stage_metrics_table
            else:
                log_error("HDF5StatsLogger", f"Unknown table: {table_name}")
                raise ValueError(f"Unknown table: {table_name}")
            
            df = pd.DataFrame(table.read())
            
            log_success("HDF5StatsLogger", f"Created DataFrame with {len(df)} rows")
            log_data("HDF5StatsLogger", "DataFrame shape", df.shape)
            
            return df
    
    # ==================== Context Manager ====================

    def __enter__(self):
        """Context manager entry."""
        log_debug("HDF5StatsLogger", "Entering context manager")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup."""
        log_debug("HDF5StatsLogger", "Exiting context manager")
        self.close()
        return False

    def close(self):
        """Close HDF5 file properly and release file lock."""
        # Close HDF5 file
        if hasattr(self, 'h5file') and self.h5file.isopen:
            log_info("HDF5StatsLogger", "Closing HDF5 file")
            self.h5file.close()
            log_success("HDF5StatsLogger", "HDF5 file closed")

        # Release file lock
        if hasattr(self, '_file_lock') and self._file_lock._locked:
            log_debug("HDF5StatsLogger", "Releasing file lock")
            self._file_lock.release()
            log_success("HDF5StatsLogger", "File lock released")

    def __del__(self):
        """Destructor - ensure file is closed and lock is released."""
        self.close()


def create_stats_logger(
    controller_dir: str,
    algorithm: str
) -> HDF5StatsLogger:
    """
    Factory function to create stats logger with proper directory structure.
    
    Args:
        controller_dir: Controller directory (e.g., 'controllers/op3_ddpg')
        algorithm: Algorithm name ('ddpg' or 'ppo')
        
    Returns:
        HDF5StatsLogger instance
    """
    with LogFunction("StatsManager", "create_stats_logger",
                    args=(controller_dir, algorithm)):
        
        log_info("StatsManager", f"Creating stats logger for algorithm: {algorithm}")
        
        # Create runs directory
        runs_dir = os.path.join(controller_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%H.%M.%S-%d.%m.%y")
        run_dir = os.path.join(runs_dir, f"{algorithm}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Create stats subdirectory
        stats_dir = os.path.join(run_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Create HDF5 file
        hdf5_path = os.path.join(stats_dir, "training_stats.h5")
        
        log_info("StatsManager", f"Creating HDF5 file: {hdf5_path}")
        
        logger = HDF5StatsLogger(hdf5_path, mode='w')
        
        log_success("StatsManager", f"Created stats logger at: {hdf5_path}")
        
        return logger


def get_latest_run(controller_dir: str, algorithm: str) -> Optional[str]:
    """
    Get the most recent run directory for an algorithm.
    
    Args:
        controller_dir: Controller directory
        algorithm: Algorithm name
        
    Returns:
        Path to latest run directory or None
    """
    with LogFunction("StatsManager", "get_latest_run", args=(controller_dir, algorithm)):
        log_info("StatsManager", f"Getting latest run for algorithm: {algorithm}")
        
        runs_dir = os.path.join(controller_dir, "runs")
        
        if not os.path.exists(runs_dir):
            log_warning("StatsManager", f"Runs directory not found: {runs_dir}")
            return None
        
        runs = [d for d in os.listdir(runs_dir) 
                if os.path.isdir(os.path.join(runs_dir, d)) 
                and d.startswith(algorithm + "_")]
        
        if not runs:
            log_warning("StatsManager", f"No runs found for algorithm: {algorithm}")
            return None
        
        # Sort by modification time
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)))
        latest_run = os.path.join(runs_dir, runs[-1])
        
        log_info("StatsManager", f"Found {len(runs)} runs, latest: {latest_run}")
        
        return latest_run


def get_latest_checkpoint(
    run_dir: str,
    stage_id: Optional[int] = None,
    agent_id: Optional[str] = None
) -> Optional[str]:
    """
    Get the latest checkpoint in a run directory.
    
    Args:
        run_dir: Run directory
        stage_id: Optional stage filter
        agent_id: Optional agent filter
        
    Returns:
        Path to checkpoint or None
    """
    with LogFunction("StatsManager", "get_latest_checkpoint",
                    args=(run_dir,),
                    kwargs={'stage_id': stage_id, 'agent_id': agent_id}):
        
        log_info("StatsManager", f"Getting latest checkpoint in: {run_dir}")
        
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            log_warning("StatsManager", f"Checkpoints directory not found: {checkpoints_dir}")
            return None
        
        checkpoints = []
        
        for stage_subdir in os.listdir(checkpoints_dir):
            if stage_id is not None:
                if not stage_subdir.startswith(f"stage_{stage_id}"):
                    continue
            
            stage_path = os.path.join(checkpoints_dir, stage_subdir)
            if not os.path.isdir(stage_path):
                continue
            
            for agent_subdir in os.listdir(stage_path):
                if agent_id is not None:
                    if agent_subdir != agent_id:
                        continue
                
                agent_path = os.path.join(stage_path, agent_subdir)
                if not os.path.isdir(agent_path):
                    continue
                
                for f in os.listdir(agent_path):
                    if f.endswith('.pt'):
                        checkpoint_path = os.path.join(agent_path, f)
                        checkpoints.append(checkpoint_path)
                        log_debug("StatsManager", f"Found checkpoint: {checkpoint_path}")
        
        if not checkpoints:
            log_warning("StatsManager", "No checkpoints found")
            return None
        
        # Return most recent
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        latest = checkpoints[-1]
        
        log_info("StatsManager", f"Latest checkpoint: {latest}")
        
        return latest
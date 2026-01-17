#!/usr/bin/env python3
"""
Multi-Stage Statistics Manager with HDF5 Support.

Provides comprehensive logging for multi-agent, multi-stage evolutionary training
using PyTables for efficient HDF5 storage.
"""

import os
import json
import numpy as np
import tables
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


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
    """
    
    def __init__(
        self,
        filepath: str,
        mode: str = 'w',
        complevel: int = 9,
        complib: str = 'blosc'
    ):
        """
        Initialize HDF5 statistics logger.
        
        Args:
            filepath: Path to HDF5 file
            mode: 'w' (write), 'a' (append), 'r' (read)
            complevel: Compression level (0-9)
            complib: Compression library
        """
        self.filepath = filepath
        self.mode = mode
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
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
    
    def _init_tables(self):
        """Initialize all required HDF5 tables."""
        
        # Stages table
        if 'stages' not in self.h5file.root:
            self.stages_table = self.h5file.create_table(
                self.h5file.root,
                'stages',
                STAGES_TABLE_DESC,
                "Training stages"
            )
        else:
            self.stages_table = self.h5file.root.stages
        
        # Episodes table
        if 'episodes' not in self.h5file.root:
            self.episodes_table = self.h5file.create_table(
                self.h5file.root,
                'episodes',
                EPISODES_TABLE_DESC,
                "Episode results"
            )
        else:
            self.episodes_table = self.h5file.root.episodes
        
        # Timesteps table
        if 'timesteps' not in self.h5file.root:
            self.timesteps_table = self.h5file.create_table(
                self.h5file.root,
                'timesteps',
                TIMESTEPS_TABLE_DESC,
                "Per-timestep data"
            )
        else:
            self.timesteps_table = self.h5file.root.timesteps
        
        # Agents table
        if 'agents' not in self.h5file.root:
            self.agents_table = self.h5file.create_table(
                self.h5file.root,
                'agents',
                AGENTS_TABLE_DESC,
                "Agent information"
            )
        else:
            self.agents_table = self.h5file.root.agents
        
        # Stage metrics table
        if 'stage_metrics' not in self.h5file.root:
            self.stage_metrics_table = self.h5file.create_table(
                self.h5file.root,
                'stage_metrics',
                STAGE_METRICS_TABLE_DESC,
                "Aggregated stage metrics"
            )
        else:
            self.stage_metrics_table = self.h5file.root.stage_metrics
    
    def _load_counters(self):
        """Load existing counters from data."""
        try:
            # Get max episode ID
            if len(self.episodes_table) > 0:
                self._global_episode_counter = self.episodes_table.cols.global_episode_id[-1] + 1
            
            # Get max timestep ID
            if len(self.timesteps_table) > 0:
                self._global_timestep_counter = self.timesteps_table.cols.global_timestep[-1] + 1
            
            # Get current stage
            if len(self.stages_table) > 0:
                self._current_stage_id = self.stages_table.cols.stage_id[-1]
        except Exception:
            pass
    
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
        
        return len(self.stages_table) - 1
    
    def update_stage(
        self,
        stage_id: int,
        end_episode_global: Optional[int] = None,
        status: Optional[str] = None,
        metrics: Optional[Dict] = None
    ):
        """Update stage information (e.g., on completion)."""
        # In PyTables, we need to use modify_rows or append a new row
        # Since row.update() doesn't exist, we'll read all data, modify, and rewrite
        
        if not self.stages_table.row:
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
        for i, data in enumerate(stages_data):
            if data['stage_id'] == stage_id:
                if end_episode_global is not None:
                    data['end_episode_global'] = end_episode_global
                if status is not None:
                    data['status'] = status
                if metrics is not None:
                    data['metrics'] = json.dumps(metrics)
                break
        else:
            return  # Stage not found
        
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
        if dones is None:
            dones = [False] * len(rewards)
            dones[-1] = True
        
        start_idx = self._global_timestep_counter
        
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
        
        return start_idx, end_idx
    
    # ==================== Agent Methods ====================
    
    def log_agent(
        self,
        agent_info: AgentInfo
    ):
        """Log agent information."""
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
        row = self.stage_metrics_table.row
        
        row['stage_id'] = stage_id
        row['metric_name'] = metric_name
        row['window_size'] = window_size
        row['value'] = value
        row['timestamp'] = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row.append()
        self.stage_metrics_table.flush()
    
    def log_stage_summary(
        self,
        stage_id: int,
        mean_reward: float,
        success_rate: float,
        mean_episode_length: float,
        window_size: int = 100
    ):
        """Log complete stage summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.log_stage_metric(stage_id, 'mean_reward', mean_reward, window_size, timestamp)
        self.log_stage_metric(stage_id, 'success_rate', success_rate, window_size, timestamp)
        self.log_stage_metric(stage_id, 'mean_episode_length', mean_episode_length, window_size, timestamp)
    
    # ==================== Query Methods ====================
    
    def get_stage_stats(self, stage_id: int) -> Dict:
        """Get statistics for a specific stage."""
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
        
        return stage_data
    
    def get_agent_lineage(self, agent_id: str) -> List[Dict]:
        """Trace lineage of an agent back to ancestors."""
        lineage = []
        
        current_id = agent_id
        while current_id:
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
        
        return lineage
    
    def get_all_agents_in_stage(self, stage_id: int) -> List[str]:
        """Get all unique agent IDs in a stage."""
        agents = set()
        
        for ep in self.episodes_table.where(f'stage_id == {stage_id}'):
            agents.add(ep['agent_id'])
        
        return list(agents)
    
    def get_episodes_for_agent(
        self,
        agent_id: str,
        stage_id: Optional[int] = None
    ) -> List[Dict]:
        """Get all episodes for a specific agent."""
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
        
        return episodes
    
    def get_best_agent_in_stage(
        self,
        stage_id: int,
        metric: str = 'total_reward'
    ) -> Tuple[Optional[str], float]:
        """Get best performing agent in a stage."""
        agent_scores = {}
        
        for ep in self.episodes_table.where(f'stage_id == {stage_id}'):
            agent_id = ep['agent_id']
            score = ep[metric]
            
            if agent_id not in agent_scores:
                agent_scores[agent_id] = []
            agent_scores[agent_id].append(score)
        
        if not agent_scores:
            return None, 0.0
        
        # Average score per agent
        agent_avgs = {k: np.mean(v) for k, v in agent_scores.items()}
        
        best_agent = max(agent_avgs, key=agent_avgs.get)
        best_score = agent_avgs[best_agent]
        
        return best_agent, best_score
    
    def get_progress(self) -> Dict:
        """Get overall training progress."""
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
        
        return progress
    
    # ==================== Export Methods ====================
    
    def export_episodes_csv(self, filepath: str, stage_id: Optional[int] = None):
        """Export episodes to CSV format."""
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
    
    def get_dataframe(self, table_name: str):
        """Get table as pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError:
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
            raise ValueError(f"Unknown table: {table_name}")
        
        return pd.DataFrame(table.read())
    
    # ==================== Context Manager ====================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup."""
        self.close()
        return False
    
    def close(self):
        """Close HDF5 file properly."""
        if hasattr(self, 'h5file') and self.h5file.isopen:
            self.h5file.close()
    
    def __del__(self):
        """Destructor - ensure file is closed."""
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
    
    return HDF5StatsLogger(hdf5_path, mode='w')


def get_latest_run(controller_dir: str, algorithm: str) -> Optional[str]:
    """
    Get the most recent run directory for an algorithm.
    
    Args:
        controller_dir: Controller directory
        algorithm: Algorithm name
        
    Returns:
        Path to latest run directory or None
    """
    runs_dir = os.path.join(controller_dir, "runs")
    
    if not os.path.exists(runs_dir):
        return None
    
    runs = [d for d in os.listdir(runs_dir) 
            if os.path.isdir(os.path.join(runs_dir, d)) 
            and d.startswith(algorithm + "_")]
    
    if not runs:
        return None
    
    # Sort by modification time
    runs.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)))
    
    return os.path.join(runs_dir, runs[-1])


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
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    
    if not os.path.exists(checkpoints_dir):
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
                    checkpoints.append(os.path.join(agent_path, f))
    
    if not checkpoints:
        return None
    
    # Return most recent
    checkpoints.sort(key=lambda x: os.path.getmtime(x))
    return checkpoints[-1]


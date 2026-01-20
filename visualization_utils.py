"""
Multi-Stage Visualization Utilities.

Provides plotting and visualization for multi-agent, multi-stage evolutionary training:
- Stage-aware training curves
- Cross-stage performance comparison
- Agent lineage tracking
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import networkx as nx
from datetime import datetime

# Try to import pandas for data handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class MultiStagePlotter:
    """
    Plot training metrics with stage awareness.
    
    Supports:
    - Rewards by stage
    - Success rates by stage
    - Episode lengths by stage
    - Combined learning curves
    """
    
    # Color palette for stages
    STAGE_COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]
    
    def __init__(self, hdf5_path: str):
        """
        Initialize plotter with HDF5 statistics file.
        
        Args:
            hdf5_path: Path to training_stats.h5
        """
        self.hdf5_path = hdf5_path
        self.data = self._load_data()
        self.stages = self._get_stages()
        
        # Setup plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from HDF5 file."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for visualization")
        
        import tables
        
        data = {}
        
        with tables.open_file(self.hdf5_path, mode='r') as h5file:
            # Load stages
            data['stages'] = pd.DataFrame(h5file.root.stages.read())
            
            # Load episodes
            data['episodes'] = pd.DataFrame(h5file.root.episodes.read())
            
            # Load stage metrics
            if hasattr(h5file.root, 'stage_metrics'):
                data['stage_metrics'] = pd.DataFrame(h5file.root.stage_metrics.read())
            
            # Load agents
            if hasattr(h5file.root, 'agents'):
                data['agents'] = pd.DataFrame(h5file.root.agents.read())
        
        return data
    
    def _get_stages(self) -> List[Dict]:
        """Get stage information."""
        if 'stages' not in self.data:
            return []
        
        stages = []
        for _, row in self.data['stages'].iterrows():
            stages.append({
                'stage_id': int(row['stage_id']),
                'stage_name': row['stage_name'],
                'start_episode': int(row['start_episode_global']),
                'end_episode': int(row['end_episode_global']) if row['end_episode_global'] >= 0 else None,
                'status': row['status']
            })
        
        return stages
    
    def _get_stage_color(self, stage_id: int) -> str:
        """Get color for a stage."""
        return self.STAGE_COLORS[stage_id % len(self.STAGE_COLORS)]
    
    def plot_rewards_by_stage(
        self,
        save_path: Optional[str] = None,
        show_moving_avg: bool = True,
        window_size: int = 50
    ) -> plt.Figure:
        """
        Plot episode rewards colored by stage.

        Args:
            save_path: Optional path to save figure
            show_moving_avg: Whether to show moving average
            window_size: Window size for moving average

        Returns:
            Matplotlib figure
        """
        if 'episodes' not in self.data or len(self.data['episodes']) == 0:
            print("No episode data found")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        episodes_df = self.data['episodes']
        has_data = False

        # Plot each stage with different color
        for stage in self.stages:
            stage_id = stage['stage_id']
            start_ep = stage['start_episode']
            end_ep = stage['end_episode'] or episodes_df['global_episode_id'].max()

            # Filter episodes for this stage
            mask = (episodes_df['global_episode_id'] >= start_ep) & \
                   (episodes_df['global_episode_id'] <= end_ep)
            stage_episodes = episodes_df[mask]

            if len(stage_episodes) == 0:
                continue

            has_data = True

            # Plot raw rewards
            ax.scatter(
                stage_episodes['global_episode_id'],
                stage_episodes['total_reward'],
                c=[self._get_stage_color(stage_id)] * len(stage_episodes),
                alpha=0.5,
                s=20,
                label=f"Stage {stage_id}: {stage['stage_name']}"
            )

            # Plot moving average
            if show_moving_avg and len(stage_episodes) > window_size:
                rewards = stage_episodes['total_reward'].values
                ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ma_x = stage_episodes['global_episode_id'].values[window_size-1:]

                ax.plot(
                    ma_x, ma,
                    color=self._get_stage_color(stage_id),
                    linewidth=2,
                    linestyle='-'
                )

        if not has_data:
            ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Episode Rewards by Stage (No Data)')
        else:
            ax.set_xlabel('Global Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title('Episode Rewards by Stage')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
    
    def plot_success_rates_by_stage(
        self,
        save_path: Optional[str] = None,
        window_size: int = 100
    ) -> plt.Figure:
        """
        Plot success rates by stage.

        Args:
            save_path: Optional path to save figure
            window_size: Window size for rolling average

        Returns:
            Matplotlib figure
        """
        if 'episodes' not in self.data or len(self.data['episodes']) == 0:
            print("No episode data found")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        episodes_df = self.data['episodes']
        has_data = False

        # Calculate cumulative success rate
        episodes_df = episodes_df.copy()
        episodes_df['cumulative_success'] = episodes_df['success'].expanding().mean()

        # Plot each stage
        for stage in self.stages:
            stage_id = stage['stage_id']
            start_ep = stage['start_episode']
            end_ep = stage['end_episode'] or episodes_df['global_episode_id'].max()

            mask = (episodes_df['global_episode_id'] >= start_ep) & \
                   (episodes_df['global_episode_id'] <= end_ep)
            stage_episodes = episodes_df[mask]

            if len(stage_episodes) == 0:
                continue

            has_data = True

            ax.plot(
                stage_episodes['global_episode_id'],
                stage_episodes['cumulative_success'],
                color=self._get_stage_color(stage_id),
                linewidth=2,
                label=f"Stage {stage_id}: {stage['stage_name']}"
            )

        if not has_data:
            ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Success Rate by Stage (No Data)')
        else:
            ax.set_xlabel('Global Episode')
            ax.set_ylabel('Cumulative Success Rate')
            ax.set_title('Success Rate by Stage')
            ax.legend(loc='lower right')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
    
    def plot_episode_lengths_by_stage(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot episode lengths by stage.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if 'episodes' not in self.data or len(self.data['episodes']) == 0:
            print("No episode data found")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        episodes_df = self.data['episodes']

        # Calculate mean episode length per stage
        stage_lengths = []
        stage_names = []

        for stage in self.stages:
            stage_id = stage['stage_id']
            start_ep = stage['start_episode']
            end_ep = stage['end_episode'] or episodes_df['global_episode_id'].max()

            mask = (episodes_df['global_episode_id'] >= start_ep) & \
                   (episodes_df['global_episode_id'] <= end_ep)
            stage_episodes = episodes_df[mask]

            if len(stage_episodes) == 0:
                continue

            mean_length = stage_episodes['total_steps'].mean()
            std_length = stage_episodes['total_steps'].std()

            stage_lengths.append((mean_length, std_length))
            stage_names.append(f"S{stage_id}: {stage['stage_name']}")

        if not stage_lengths:
            ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Episode Length by Stage (No Data)')
        else:
            # Create bar chart
            x = np.arange(len(stage_names))
            means = [m for m, _ in stage_lengths]
            stds = [s for _, s in stage_lengths]
            colors = [self._get_stage_color(i) for i in range(len(stage_names))]

            bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

            ax.set_xlabel('Stage')
            ax.set_ylabel('Mean Episode Length')
            ax.set_title('Episode Length by Stage')
            ax.set_xticks(x)
            ax.set_xticklabels(stage_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
    
    def plot_learning_curves(
        self,
        save_path: Optional[str] = None,
        window_size: int = 50
    ) -> plt.Figure:
        """
        Plot combined learning curves (rewards, success rate, length).

        Args:
            save_path: Optional path to save figure
            window_size: Window size for smoothing

        Returns:
            Matplotlib figure
        """
        if 'episodes' not in self.data or len(self.data['episodes']) == 0:
            print("No episode data found")
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        episodes_df = self.data['episodes']
        has_data = len(episodes_df) > 0

        if not has_data:
            for ax in axes:
                ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            axes[0].set_title('Learning Curves (No Data)')
            axes[2].set_xlabel('Global Episode')
        else:
            # Plot 1: Rewards
            ax1 = axes[0]
            ax1.plot(
                episodes_df['global_episode_id'],
                episodes_df['total_reward'],
                alpha=0.3,
                color='blue'
            )

            if len(episodes_df) > window_size:
                ma = episodes_df['total_reward'].rolling(window=window_size).mean()
                ax1.plot(
                    episodes_df['global_episode_id'],
                    ma,
                    color='blue',
                    linewidth=2,
                    label=f'{window_size}-episode MA'
                )

            ax1.set_ylabel('Total Reward')
            ax1.set_title('Learning Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add stage boundaries
            for stage in self.stages:
                if stage['end_episode']:
                    ax1.axvline(x=stage['end_episode'], color='gray', linestyle='--', alpha=0.5)

            # Plot 2: Success Rate
            ax2 = axes[1]
            success_rate = episodes_df['success'].rolling(window=window_size).mean()
            ax2.plot(
                episodes_df['global_episode_id'],
                success_rate,
                color='green',
                linewidth=2
            )
            ax2.set_ylabel('Success Rate')
            ax2.set_ylim([0, 1.05])
            ax2.grid(True, alpha=0.3)

            # Add stage boundaries
            for stage in self.stages:
                if stage['end_episode']:
                    ax2.axvline(x=stage['end_episode'], color='gray', linestyle='--', alpha=0.5)

            # Plot 3: Episode Length
            ax3 = axes[2]
            ax3.plot(
                episodes_df['global_episode_id'],
                episodes_df['total_steps'],
                alpha=0.3,
                color='orange'
            )

            if len(episodes_df) > window_size:
                ma = episodes_df['total_steps'].rolling(window=window_size).mean()
                ax3.plot(
                    episodes_df['global_episode_id'],
                    ma,
                    color='orange',
                    linewidth=2,
                    label=f'{window_size}-episode MA'
                )

            ax3.set_xlabel('Global Episode')
            ax3.set_ylabel('Episode Length')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add stage boundaries
            for stage in self.stages:
                if stage['end_episode']:
                    ax3.axvline(x=stage['end_episode'], color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
    
    def plot_stage_comparison(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comparison plot across stages.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if 'episodes' not in self.data or 'stages' not in self.data or len(self.data['episodes']) == 0:
            print("Required data not found")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        episodes_df = self.data['episodes']

        # Collect stage statistics
        stage_stats = []
        for stage in self.stages:
            stage_id = stage['stage_id']
            start_ep = stage['start_episode']
            end_ep = stage['end_episode'] or episodes_df['global_episode_id'].max()

            mask = (episodes_df['global_episode_id'] >= start_ep) & \
                   (episodes_df['global_episode_id'] <= end_ep)
            stage_episodes = episodes_df[mask]

            if len(stage_episodes) == 0:
                continue

            stage_stats.append({
                'name': f"S{stage_id}: {stage['stage_name']}",
                'mean_reward': stage_episodes['total_reward'].mean(),
                'std_reward': stage_episodes['total_reward'].std(),
                'success_rate': stage_episodes['success'].mean(),
                'mean_length': stage_episodes['total_steps'].mean(),
                'count': len(stage_episodes)
            })

        if not stage_stats:
            for ax in axes.flatten():
                ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Stage Comparison (No Data)')
            return fig

        names = [s['name'] for s in stage_stats]
        colors = [self._get_stage_color(i) for i in range(len(stage_stats))]

        # Plot 1: Mean Reward
        ax1 = axes[0, 0]
        means = [s['mean_reward'] for s in stage_stats]
        stds = [s['std_reward'] for s in stage_stats]
        bars = ax1.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Mean Reward by Stage')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Success Rate
        ax2 = axes[0, 1]
        success_rates = [s['success_rate'] for s in stage_stats]
        bars = ax2.bar(names, success_rates, color=colors, alpha=0.7)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate by Stage')
        ax2.set_ylim([0, 1.05])
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Episode Count
        ax3 = axes[1, 0]
        counts = [s['count'] for s in stage_stats]
        bars = ax3.bar(names, counts, color=colors, alpha=0.7)
        ax3.set_ylabel('Episode Count')
        ax3.set_title('Episodes per Stage')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Mean Episode Length
        ax4 = axes[1, 1]
        lengths = [s['mean_length'] for s in stage_stats]
        bars = ax4.bar(names, lengths, color=colors, alpha=0.7)
        ax4.set_ylabel('Mean Episode Length')
        ax4.set_title('Mean Episode Length by Stage')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
    
    def generate_summary_report(self) -> Dict:
        """Generate summary statistics for the training run."""
        if 'episodes' not in self.data:
            return {}
        
        episodes_df = self.data['episodes']
        
        summary = {
            'total_episodes': len(episodes_df),
            'total_successful': int(episodes_df['success'].sum()),
            'overall_success_rate': float(episodes_df['success'].mean()),
            'mean_reward': float(episodes_df['total_reward'].mean()),
            'std_reward': float(episodes_df['total_reward'].std()),
            'mean_episode_length': float(episodes_df['total_steps'].mean()),
            'stages_completed': len([s for s in self.stages if s['status'] == 'completed']),
            'stage_details': []
        }
        
        # Per-stage details
        for stage in self.stages:
            stage_id = stage['stage_id']
            start_ep = stage['start_episode']
            end_ep = stage['end_episode'] or episodes_df['global_episode_id'].max()
            
            mask = (episodes_df['global_episode_id'] >= start_ep) & \
                   (episodes_df['global_episode_id'] <= end_ep)
            stage_episodes = episodes_df[mask]
            
            if len(stage_episodes) == 0:
                continue
            
            summary['stage_details'].append({
                'stage_id': stage_id,
                'stage_name': stage['stage_name'],
                'episodes': len(stage_episodes),
                'mean_reward': float(stage_episodes['total_reward'].mean()),
                'success_rate': float(stage_episodes['success'].mean())
            })
        
        return summary


class AgentLineageTracker:
    """
    Track and visualize agent lineage (family tree).
    
    Shows which agents descended from which parents across generations.
    """
    
    def __init__(self, hdf5_path: str):
        """
        Initialize lineage tracker.
        
        Args:
            hdf5_path: Path to training_stats.h5
        """
        self.hdf5_path = hdf5_path
        self.agents = self._load_agents()
    
    def _load_agents(self) -> List[Dict]:
        """Load agent data from HDF5."""
        agents = []
        
        try:
            import tables
            
            with tables.open_file(self.hdf5_path, mode='r') as h5file:
                if hasattr(h5file.root, 'agents'):
                    agents_df = pd.DataFrame(h5file.root.agents.read())
                    agents = agents_df.to_dict('records')
        except Exception as e:
            print(f"Error loading agents: {e}")
        
        return agents
    
    def build_lineage_graph(self) -> nx.DiGraph:
        """
        Build a directed graph of agent lineage.
        
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        for agent in self.agents:
            agent_id = agent.get('agent_id', 'unknown')
            parent_id = agent.get('parent_id')
            stage_id = agent.get('stage_id', 0)
            
            G.add_node(
                agent_id,
                stage_id=stage_id,
                lineage_depth=agent.get('lineage_depth', 0)
            )
            
            if parent_id:
                G.add_edge(parent_id, agent_id)
        
        return G
    
    def plot_lineage(
        self,
        save_path: Optional[str] = None,
        max_depth: int = 5
    ) -> plt.Figure:
        """
        Plot agent lineage as a tree.
        
        Args:
            save_path: Optional path to save figure
            max_depth: Maximum depth to show
            
        Returns:
            Matplotlib figure
        """
        G = self.build_lineage_graph()
        
        if len(G.nodes) == 0:
            print("No agent data found")
            return None
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Filter by depth
        nodes_to_show = [
            n for n in G.nodes() 
            if G.nodes[n].get('lineage_depth', 0) <= max_depth
        ]
        G = G.subgraph(nodes_to_show).copy()
        
        # Create layout
        try:
            # Group by depth
            depth_groups = defaultdict(list)
            for node in G.nodes():
                depth = G.nodes[node].get('lineage_depth', 0)
                depth_groups[depth].append(node)
            
            # Create hierarchical layout
            pos = {}
            max_depth = max(depth_groups.keys()) if depth_groups else 0
            
            for depth, nodes in depth_groups.items():
                n_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    x = (i - n_nodes / 2) / (n_nodes + 1) * 10
                    y = -depth
                    pos[node] = (x, y)
            
            # Color by stage
            stage_colors = plt.cm.tab10
            node_colors = [
                stage_colors(G.nodes[n].get('stage_id', 0) % 10) 
                for n in G.nodes()
            ]
            
            # Draw graph
            nx.draw(
                G, pos,
                ax=ax,
                with_labels=True,
                node_color=node_colors,
                node_size=500,
                font_size=8,
                arrows=True,
                arrowsize=10,
                edge_color='gray',
                alpha=0.8
            )
            
            # Add legend for stages
            unique_stages = set(G.nodes[n].get('stage_id', 0) for n in G.nodes())
            legend_patches = [
                mpatches.Patch(color=stage_colors(s % 10), label=f'Stage {s}')
                for s in sorted(unique_stages)
            ]
            ax.legend(handles=legend_patches, loc='upper right')
            
        except Exception as e:
            print(f"Error plotting lineage: {e}")
            ax.text(0.5, 0.5, "Could not generate lineage plot", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Agent Lineage (Parent → Child)')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def trace_lineage(self, agent_id: str) -> List[Dict]:
        """
        Trace lineage of a specific agent back to ancestors.
        
        Args:
            agent_id: Target agent ID
            
        Returns:
            List of ancestor agents (from oldest to newest)
        """
        lineage = []
        current_id = agent_id
        
        while current_id:
            # Find agent
            found = None
            for agent in self.agents:
                if agent.get('agent_id') == current_id:
                    found = agent
                    break
            
            if found:
                lineage.append(found)
                current_id = found.get('parent_id')
            else:
                break
        
        return lineage
    
    def get_generation_counts(self) -> Dict[int, int]:
        """Count agents per generation/depth."""
        counts = defaultdict(int)
        
        for agent in self.agents:
            depth = agent.get('lineage_depth', 0)
            counts[depth] += 1
        
        return dict(counts)


class PerformanceComparator:
    """
    Compare performance across stages and agents.
    """
    
    def __init__(self, hdf5_path: str):
        """
        Initialize comparator.
        
        Args:
            hdf5_path: Path to training_stats.h5
        """
        self.hdf5_path = hdf5_path
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load data from HDF5."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        
        import tables
        
        data = {}
        
        with tables.open_file(self.hdf5_path, mode='r') as h5file:
            data['stages'] = pd.DataFrame(h5file.root.stages.read())
            data['episodes'] = pd.DataFrame(h5file.root.episodes.read())
            
            if hasattr(h5file.root, 'agents'):
                data['agents'] = pd.DataFrame(h5file.root.agents.read())
            
            if hasattr(h5file.root, 'stage_metrics'):
                data['stage_metrics'] = pd.DataFrame(h5file.root.stage_metrics.read())
        
        return data
    
    def compare_stages(self, metric: str = 'total_reward') -> Dict:
        """
        Compare stages on a specific metric.
        
        Args:
            metric: Metric to compare ('total_reward', 'success', 'total_steps')
            
        Returns:
            Comparison dictionary
        """
        if 'episodes' not in self.data or 'stages' not in self.data:
            return {}
        
        episodes_df = self.data['episodes']
        stages_df = self.data['stages']
        
        comparison = {}
        
        for _, stage in stages_df.iterrows():
            stage_id = int(stage['stage_id'])
            stage_name = stage['stage_name']
            
            start_ep = int(stage['start_episode_global'])
            end_ep = int(stage['end_episode_global'])
            
            if end_ep < 0:
                end_ep = episodes_df['global_episode_id'].max()
            
            mask = (episodes_df['global_episode_id'] >= start_ep) & \
                   (episodes_df['global_episode_id'] <= end_ep)
            stage_episodes = episodes_df[mask]
            
            if len(stage_episodes) == 0:
                continue
            
            if metric == 'success':
                value = float(stage_episodes['success'].mean())
            else:
                value = float(stage_episodes[metric].mean())
            
            comparison[f"Stage {stage_id}: {stage_name}"] = {
                'value': value,
                'count': len(stage_episodes),
                'std': float(stage_episodes[metric].std()) if len(stage_episodes) > 1 else 0
            }
        
        return comparison
    
    def compare_agents_in_stage(
        self,
        stage_id: int,
        metric: str = 'total_reward'
    ) -> Dict[str, Dict]:
        """
        Compare agents within a stage.
        
        Args:
            stage_id: Stage to analyze
            metric: Metric to compare
            
        Returns:
            Agent comparison dictionary
        """
        if 'episodes' not in self.data:
            return {}
        
        episodes_df = self.data['episodes']
        
        mask = episodes_df['stage_id'] == stage_id
        stage_episodes = episodes_df[mask]
        
        if len(stage_episodes) == 0:
            return {}
        
        # Group by agent
        agent_stats = stage_episodes.groupby('agent_id')[metric].agg(['mean', 'std', 'count'])
        
        comparison = {}
        for agent_id, stats in agent_stats.iterrows():
            comparison[agent_id] = {
                'value': float(stats['mean']),
                'std': float(stats['std']) if not pd.isna(stats['std']) else 0,
                'episodes': int(stats['count'])
            }
        
        return comparison
    
    def get_best_agent_per_stage(self) -> Dict[int, Dict]:
        """
        Get best performing agent for each stage.
        
        Returns:
            Dictionary mapping stage_id to best agent info
        """
        best_agents = {}
        
        for _, stage in self.data['stages'].iterrows():
            stage_id = int(stage['stage_id'])
            
            comparison = self.compare_agents_in_stage(stage_id)
            
            if not comparison:
                continue
            
            # Find best
            best_agent = max(comparison.keys(), key=lambda k: comparison[k]['value'])
            
            best_agents[stage_id] = {
                'agent_id': best_agent,
                'value': comparison[best_agent]['value'],
                'std': comparison[best_agent]['std']
            }
        
        return best_agents


def visualize_run(
    run_dir: str,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate all visualizations for a training run.
    
    Args:
        run_dir: Run directory containing stats/training_stats.h5
        output_dir: Output directory for plots (default: run_dir/plots)
        
    Returns:
        Dictionary with paths to generated plots
    """
    hdf5_path = os.path.join(run_dir, "stats", "training_stats.h5")
    
    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found: {hdf5_path}")
        return {}
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    generated = {}
    
    try:
        # Initialize plotters
        plotter = MultiStagePlotter(hdf5_path)
        lineage_tracker = AgentLineageTracker(hdf5_path)
        comparator = PerformanceComparator(hdf5_path)
        
        # Generate plots
        fig = plotter.plot_rewards_by_stage()
        if fig:
            path = os.path.join(output_dir, "rewards_by_stage.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated['rewards_by_stage'] = path
            plt.close(fig)
        
        fig = plotter.plot_success_rates_by_stage()
        if fig:
            path = os.path.join(output_dir, "success_rates_by_stage.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated['success_rates_by_stage'] = path
            plt.close(fig)
        
        fig = plotter.plot_episode_lengths_by_stage()
        if fig:
            path = os.path.join(output_dir, "episode_lengths_by_stage.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated['episode_lengths_by_stage'] = path
            plt.close(fig)
        
        fig = plotter.plot_learning_curves()
        if fig:
            path = os.path.join(output_dir, "learning_curves.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated['learning_curves'] = path
            plt.close(fig)
        
        fig = plotter.plot_stage_comparison()
        if fig:
            path = os.path.join(output_dir, "stage_comparison.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated['stage_comparison'] = path
            plt.close(fig)
        
        # Lineage plot
        fig = lineage_tracker.plot_lineage()
        if fig:
            path = os.path.join(output_dir, "agent_lineage.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            generated['agent_lineage'] = path
            plt.close(fig)
        
        # Generate summary
        summary = plotter.generate_summary_report()
        summary_path = os.path.join(output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        generated['summary'] = summary_path
        
        print(f"Generated {len(generated)} visualizations in {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    return generated


# Import pandas for data handling
try:
    import pandas as pd
except ImportError:
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument('run_dir', help="Directory containing training_stats.h5")
    parser.add_argument('--output', '-o', help="Output directory for plots")
    
    args = parser.parse_args()
    
    generated = visualize_run(args.run_dir, args.output)
    
    print("\nGenerated files:")
    for name, path in generated.items():
        print(f"  {name}: {path}")


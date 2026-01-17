# graph_interpolator_enhanced.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import pandas as pd
import csv
import threading
import time
from collections import deque

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack elements
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel for scrolling
        self.bind_mousewheel()
        
        # Update scroll region on frame resize
        self.scrollable_frame.bind("<Configure>", self.update_scrollregion)
        self.canvas.bind("<Configure>", self.update_window_width)
    
    def bind_mousewheel(self):
        """Bind mousewheel for scrolling"""
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind_all("<Button-5>", self.on_mousewheel)  # Linux scroll down
    
    def unbind_mousewheel(self):
        """Unbind mousewheel"""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        # Windows/MacOS
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
    
    def update_scrollregion(self, event=None):
        """Update scroll region when frame changes size"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def update_window_width(self, event=None):
        """Update scrollable frame width when canvas changes size"""
        self.canvas.itemconfig(self.window_id, width=event.width)

class GraphInterpolator:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Interpolator Pro")
        self.root.geometry("1400x800")
        
        # Data storage
        self.original_data = {}  # Store original unsmoothed data
        self.data_series = {}  # {series_name: {'x': [], 'y': [], 'color': '', 'style': '', 'visible': True}}
        self.current_series = None
        self.interpolation_methods = ['linear', 'cubic', 'quadratic', 'nearest']
        self.selected_method = 'cubic'
        
        # Color and line style options
        self.colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
        
        # Smoothness control
        self.smoothness_level = 1  # 1-10 scale
        self.max_points_per_series = 1000  # Maximum points per series at max smoothness
        self.slider_update_delay = 300  # ms delay before applying smoothness
        self.slider_timer = None
        self.slider_value = 1
        
        # Custom labels and title
        self.x_label = "X"
        self.y_label = "Y"
        self.plot_title = "Multi-Series Graph"
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable control frame
        self.control_frame = ScrollableFrame(main_frame, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        # Configure scrollable frame width
        self.control_frame.scrollable_frame.configure(width=400)  # Fixed width for sidebar
        
        graph_frame = ttk.Frame(main_frame, padding="10")
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Get the scrollable inner frame
        inner_frame = self.control_frame.scrollable_frame
        
        # Control Panel
        ttk.Label(inner_frame, text="Multi-Series Graph", 
                 font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=(0, 15), columnspan=2, sticky=tk.W)
        
        # Data series management section
        series_section = ttk.LabelFrame(inner_frame, text="Data Series", padding="10")
        series_section.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Frame for series checkboxes (with fixed height)
        self.series_list_frame = ttk.Frame(series_section)
        self.series_list_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configure series list frame
        self.series_list_frame.grid_columnconfigure(0, weight=1)
        
        # Load CSV button
        ttk.Button(series_section, text="Load CSV", command=self.load_csv,
                  width=15).grid(row=1, column=0, padx=2, pady=10, sticky=tk.W)
        ttk.Button(series_section, text="Clear All", command=self.clear_all_series,
                  width=15).grid(row=1, column=1, padx=2, pady=10, sticky=tk.E)
        
        # Smoothness control section
        smoothness_section = ttk.LabelFrame(inner_frame, text="Data Smoothness", padding="10")
        smoothness_section.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(smoothness_section, text="Smoothness Level:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Smoothness slider
        self.smoothness_var = tk.DoubleVar(value=self.smoothness_level)
        self.smoothness_slider = ttk.Scale(
            smoothness_section,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.smoothness_var,
            length=200,
            command=self.on_smoothness_change
        )
        self.smoothness_slider.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Smoothness value display
        self.smoothness_value_label = ttk.Label(smoothness_section, text=f"Level: {int(self.smoothness_level)}")
        self.smoothness_value_label.grid(row=2, column=0, sticky=tk.W)
        
        # Info labels
        ttk.Label(smoothness_section, text="Low = More Detail", font=('Arial', 8), 
                 foreground="blue").grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(smoothness_section, text="High = More Smooth", font=('Arial', 8), 
                 foreground="green").grid(row=2, column=2, sticky=tk.W, padx=(10, 0))
        
        # Current point count display
        self.points_info_label = ttk.Label(smoothness_section, text="Points: -", font=('Arial', 9))
        self.points_info_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Series management section
        management_section = ttk.LabelFrame(inner_frame, text="Series Management", padding="10")
        management_section.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Point management for selected series
        ttk.Label(management_section, text="Selected Series:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        self.series_var = tk.StringVar()
        self.series_combo = ttk.Combobox(management_section, textvariable=self.series_var, 
                                        state="readonly", width=20)
        self.series_combo.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        self.series_combo.bind('<<ComboboxSelected>>', self.on_series_selected)
        
        # Point management buttons
        ttk.Button(management_section, text="Add Point", command=self.add_point,
                  width=12).grid(row=2, column=0, padx=2, pady=5)
        ttk.Button(management_section, text="Edit Point", command=self.edit_point,
                  width=12).grid(row=2, column=1, padx=2, pady=5)
        ttk.Button(management_section, text="Delete Point", command=self.delete_point,
                  width=12).grid(row=2, column=2, padx=2, pady=5)
        
        ttk.Button(management_section, text="Clear Series", command=self.clear_series,
                  width=12).grid(row=3, column=0, padx=2, pady=5)
        ttk.Button(management_section, text="Delete Series", command=self.delete_series,
                  width=12).grid(row=3, column=1, padx=2, pady=5)
        
        # Series color and style selection
        ttk.Label(management_section, text="Line Color:").grid(row=4, column=0, sticky=tk.W, pady=(10, 2))
        self.color_var = tk.StringVar(value="blue")
        color_combo = ttk.Combobox(management_section, textvariable=self.color_var, 
                                  values=self.colors, width=10, state="readonly")
        color_combo.grid(row=4, column=1, sticky=tk.W, pady=(10, 2))
        color_combo.bind('<<ComboboxSelected>>', self.update_series_color)
        
        ttk.Label(management_section, text="Line Style:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.style_var = tk.StringVar(value="-")
        style_combo = ttk.Combobox(management_section, textvariable=self.style_var, 
                                  values=['Solid (-)', 'Dashed (--)', 'Dash-dot (-.)', 'Dotted (:)', 
                                         'Loosely dotted', 'Dash-dot-dot'],
                                  width=15, state="readonly")
        style_combo.grid(row=5, column=1, sticky=tk.W, pady=2)
        style_combo.bind('<<ComboboxSelected>>', self.update_series_style)
        
        # Customization section
        customization_section = ttk.LabelFrame(inner_frame, text="Graph Customization", padding="10")
        customization_section.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # X Label
        ttk.Label(customization_section, text="X Label:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.x_label_var = tk.StringVar(value=self.x_label)
        x_label_entry = ttk.Entry(customization_section, textvariable=self.x_label_var, width=20)
        x_label_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        x_label_entry.bind('<KeyRelease>', lambda e: self.update_labels())
        
        # Y Label
        ttk.Label(customization_section, text="Y Label:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.y_label_var = tk.StringVar(value=self.y_label)
        y_label_entry = ttk.Entry(customization_section, textvariable=self.y_label_var, width=20)
        y_label_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        y_label_entry.bind('<KeyRelease>', lambda e: self.update_labels())
        
        # Plot Title
        ttk.Label(customization_section, text="Plot Title:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.plot_title_var = tk.StringVar(value=self.plot_title)
        title_entry = ttk.Entry(customization_section, textvariable=self.plot_title_var, width=20)
        title_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        title_entry.bind('<KeyRelease>', lambda e: self.update_labels())
        
        # Interpolation settings
        interp_section = ttk.LabelFrame(inner_frame, text="Interpolation Settings", padding="10")
        interp_section.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(interp_section, text="Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar(value=self.selected_method)
        method_combo = ttk.Combobox(interp_section, textvariable=self.method_var, 
                                   values=self.interpolation_methods, state="readonly", width=15)
        method_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        method_combo.bind('<<ComboboxSelected>>', self.on_method_change)
        
        ttk.Label(interp_section, text="Interp Points:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.interp_points_var = tk.StringVar(value="200")
        interp_entry = ttk.Entry(interp_section, textvariable=self.interp_points_var, width=15)
        interp_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(5, 0))
        interp_entry.bind('<KeyRelease>', lambda e: self.update_graph())
        
        # Show interpolation checkbox
        self.show_interp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(interp_section, text="Show Interpolation", 
                       variable=self.show_interp_var, command=self.update_graph).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Plot options
        options_section = ttk.LabelFrame(inner_frame, text="Plot Options", padding="10")
        options_section.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.show_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_section, text="Show Data Points", 
                       variable=self.show_points_var, command=self.update_graph).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_section, text="Show Grid", 
                       variable=self.show_grid_var, command=self.update_graph).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        self.show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_section, text="Show Legend", 
                       variable=self.show_legend_var, command=self.update_graph).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.show_stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_section, text="Show Statistics", 
                       variable=self.show_stats_var, command=self.toggle_statistics).grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # Action buttons
        action_frame = ttk.Frame(inner_frame)
        action_frame.grid(row=7, column=0, columnspan=2, pady=(0, 15))
        
        ttk.Button(action_frame, text="Save Image", command=self.save_image, 
                  style="Accent.TButton", width=15).grid(row=0, column=0, padx=5)
        ttk.Button(action_frame, text="Load Sample", command=self.load_sample_data, 
                  width=15).grid(row=0, column=1, padx=5)
        ttk.Button(action_frame, text="Export Data", command=self.export_data, 
                  width=15).grid(row=1, column=0, padx=5, pady=(10, 0))
        ttk.Button(action_frame, text="Copy to CSV", command=self.copy_to_csv, 
                  width=15).grid(row=1, column=1, padx=5, pady=(10, 0))
        
        # Statistics display (initially hidden)
        self.stats_section = ttk.LabelFrame(inner_frame, text="Statistics", padding="10")
        self.stats_section.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.stats_section.grid_remove()  # Hide initially
        
        self.stats_text = tk.Text(self.stats_section, height=8, width=35, font=('Courier', 9))
        scrollbar_stats = ttk.Scrollbar(self.stats_section, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar_stats.set)
        
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_stats.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.stats_section.grid_columnconfigure(0, weight=1)
        self.stats_section.grid_rowconfigure(0, weight=1)
        
        # Graph area
        self.setup_graph_area(graph_frame)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Arial', 10, 'bold'))
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_graph_area(self, parent):
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_closing(self):
        """Clean up when closing window"""
        self.control_frame.unbind_mousewheel()
        self.root.destroy()
    
    def on_smoothness_change(self, value):
        """Handle smoothness slider changes with delay."""
        # Update the displayed value
        int_value = int(float(value))
        self.smoothness_value_label.config(text=f"Level: {int_value}")
        self.slider_value = int_value
        
        # Cancel any pending update
        if self.slider_timer:
            self.root.after_cancel(self.slider_timer)
        
        # Schedule update after delay
        self.slider_timer = self.root.after(self.slider_update_delay, self.apply_smoothness)
    
    def apply_smoothness(self):
        """Apply smoothness level to data and update graph."""
        self.smoothness_level = self.slider_value
        
        # Apply smoothness to all series
        for series_name in self.data_series:
            if series_name in self.original_data:
                self.apply_smoothness_to_series(series_name)
        
        # Update points info
        self.update_points_info()
        
        # Update graph
        self.update_graph()
    
    def apply_smoothness_to_series(self, series_name):
        """Apply smoothness to a specific series."""
        if series_name not in self.original_data:
            return
        
        original_x = self.original_data[series_name]['x']
        original_y = self.original_data[series_name]['y']
        
        if len(original_x) < 2:
            # Not enough data to smooth
            self.data_series[series_name]['x'] = original_x.copy()
            self.data_series[series_name]['y'] = original_y.copy()
            return
        
        # Calculate target number of points based on smoothness level
        # Level 1: keep all points (no smoothing)
        # Level 10: keep at most max_points_per_series points
        if self.smoothness_level == 1:
            # No smoothing
            target_points = len(original_x)
        else:
            # Calculate target points: linear scale from full data to max_points
            reduction_factor = (self.smoothness_level - 1) / 9.0  # 0 to 1
            max_reduction = min(self.max_points_per_series, len(original_x))
            target_points = int(len(original_x) * (1 - reduction_factor) + max_reduction * reduction_factor)
            target_points = max(2, min(target_points, len(original_x)))
        
        # If target is close to original size, just use original
        if target_points >= len(original_x) * 0.9:
            self.data_series[series_name]['x'] = original_x.copy()
            self.data_series[series_name]['y'] = original_y.copy()
            return
        
        # Apply smoothing/resampling
        self.resample_series(series_name, original_x, original_y, target_points)
    
    def resample_series(self, series_name, x_vals, y_vals, target_points):
        """Resample series to target number of points using moving average."""
        if target_points >= len(x_vals):
            # No resampling needed
            self.data_series[series_name]['x'] = x_vals.copy()
            self.data_series[series_name]['y'] = y_vals.copy()
            return
        
        # Calculate step size for uniform sampling
        step = len(x_vals) / target_points
        
        # Create indices for sampling
        indices = np.linspace(0, len(x_vals) - 1, target_points, dtype=int)
        
        # Apply moving average for extra smoothness
        if self.smoothness_level >= 7 and len(x_vals) > 100:
            # Higher smoothness levels get moving average
            window_size = max(3, int(len(x_vals) / target_points * 2))
            if window_size % 2 == 0:
                window_size += 1
            
            # Apply moving average to y values
            smoothed_y = self.moving_average(y_vals, window_size)
            
            # Sample the smoothed data
            sampled_x = [x_vals[i] for i in indices]
            sampled_y = [smoothed_y[i] for i in indices]
        else:
            # Simple uniform sampling
            sampled_x = [x_vals[i] for i in indices]
            sampled_y = [y_vals[i] for i in indices]
        
        self.data_series[series_name]['x'] = sampled_x
        self.data_series[series_name]['y'] = sampled_y
    
    def moving_average(self, data, window_size):
        """Apply moving average smoothing."""
        if window_size < 2:
            return data
        
        # Pad the data
        pad_size = window_size // 2
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
        
        # Apply convolution
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(padded_data, kernel, mode='valid')
        
        return smoothed.tolist()
    
    def update_points_info(self):
        """Update the points information display."""
        if not self.data_series:
            self.points_info_label.config(text="Points: -")
            return
        
        total_points = 0
        original_points = 0
        visible_series = [name for name, data in self.data_series.items() if data['visible']]
        
        for series_name in visible_series:
            if series_name in self.original_data:
                original_points += len(self.original_data[series_name]['x'])
            total_points += len(self.data_series[series_name]['x'])
        
        if original_points > 0 and original_points != total_points:
            reduction_pct = 100 * (1 - total_points / original_points)
            self.points_info_label.config(
                text=f"Points: {total_points:,} ({reduction_pct:.1f}% reduced)"
            )
        else:
            self.points_info_label.config(text=f"Points: {total_points:,}")
    
    def update_labels(self):
        """Update labels and title based on entry fields."""
        self.x_label = self.x_label_var.get()
        self.y_label = self.y_label_var.get()
        self.plot_title = self.plot_title_var.get()
        self.update_graph()
    
    def update_graph(self):
        """Update the graph with current settings."""
        if not self.data_series:
            self.ax.clear()
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_label)
            self.ax.set_title(self.plot_title)
            self.ax.grid(self.show_grid_var.get(), alpha=0.3)
            self.canvas.draw()
            return
        
        self.ax.clear()
        
        # Track all data for axis limits
        all_x = []
        all_y = []
        
        # Plot each visible series
        for series_name, series_data in self.data_series.items():
            if not series_data['visible']:
                continue
                
            x_vals = series_data['x']
            y_vals = series_data['y']
            
            if len(x_vals) < 2:
                continue
                
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            
            # Sort points by x for interpolation
            sorted_points = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
            x_sorted = np.array([p[0] for p in sorted_points])
            y_sorted = np.array([p[1] for p in sorted_points])
            
            # Get color and style
            color = series_data['color']
            line_style = series_data['style']
            
            # Plot interpolation if enabled
            if self.show_interp_var.get() and len(x_sorted) >= 2:
                try:
                    # Create interpolation function
                    method = self.selected_method
                    
                    if method == 'linear':
                        f = interpolate.interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                    elif method == 'cubic':
                        if len(x_sorted) < 4:
                            f = interpolate.interp1d(x_sorted, y_sorted, kind='quadratic', fill_value='extrapolate')
                        else:
                            f = interpolate.interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
                    elif method == 'quadratic':
                        if len(x_sorted) < 3:
                            f = interpolate.interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                        else:
                            f = interpolate.interp1d(x_sorted, y_sorted, kind='quadratic', fill_value='extrapolate')
                    else:  # nearest
                        f = interpolate.interp1d(x_sorted, y_sorted, kind='nearest', fill_value='extrapolate')
                    
                    # Generate interpolation points
                    num_points = int(self.interp_points_var.get())
                    x_min, x_max = min(x_sorted), max(x_sorted)
                    x_range = x_max - x_min
                    x_interp = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, num_points)
                    y_interp = f(x_interp)
                    
                    # Plot interpolation line
                    self.ax.plot(x_interp, y_interp, color=color, linestyle=line_style, 
                               linewidth=2, label=f'{series_name} ({method})', zorder=1)
                    
                except Exception as e:
                    print(f"Interpolation error for {series_name}: {e}")
                    # Fallback to just plotting points
                    self.ax.plot(x_sorted, y_sorted, color=color, linestyle=line_style, 
                               linewidth=2, label=series_name, zorder=1)
            else:
                # Just plot the line connecting points
                self.ax.plot(x_sorted, y_sorted, color=color, linestyle=line_style, 
                           linewidth=2, label=series_name, zorder=1)
            
            # Plot data points if enabled
            if self.show_points_var.get():
                # Don't show too many points if series is large
                if len(x_sorted) > 1000:
                    # Sample points for display
                    step = max(1, len(x_sorted) // 500)
                    x_display = x_sorted[::step]
                    y_display = y_sorted[::step]
                    self.ax.plot(x_display, y_display, marker='.', color=color, 
                               markersize=4, linestyle='', alpha=0.6, zorder=2)
                else:
                    self.ax.plot(x_sorted, y_sorted, marker='o', color=color, 
                               markersize=4, linestyle='', zorder=2)
                
                # Add point labels for first few points (if not too many)
                if len(sorted_points) <= 10:
                    for i, (x, y) in enumerate(sorted_points[:5]):  # Limit to 5 labels
                        self.ax.annotate(f'({x:.1f},{y:.1f})', (x, y), 
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, alpha=0.7)
        
        # Apply custom labels and title
        self.ax.set_xlabel(self.x_label, fontsize=12)
        self.ax.set_ylabel(self.y_label, fontsize=12)
        self.ax.set_title(f"{self.plot_title} (Smoothness: {self.smoothness_level})", 
                         fontsize=14, fontweight='bold', pad=15)
        
        # Add legend if enabled
        if self.show_legend_var.get():
            self.ax.legend(loc='best', fontsize=10)
        
        self.ax.grid(self.show_grid_var.get(), alpha=0.3)
        
        # Adjust plot limits
        if all_x and all_y:
            padding = 0.05
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            x_padding = (x_max - x_min) * padding if x_max != x_min else 1.0
            y_padding = (y_max - y_min) * padding if y_max != y_min else 1.0
            
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Add a subtle background color
        self.ax.set_facecolor('#f8f9fa')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update statistics if enabled
        if self.show_stats_var.get():
            self.update_statistics()
    
    def update_statistics(self):
        """Update statistics display."""
        if not self.data_series:
            self.stats_text.delete(1.0, tk.END)
            return
        
        stats = "STATISTICS BY SERIES\n"
        stats += "=" * 40 + "\n"
        stats += f"Smoothness Level: {self.smoothness_level}\n\n"
        
        for series_name, series_data in self.data_series.items():
            if not series_data['visible']:
                continue
                
            x_vals = series_data['x']
            y_vals = series_data['y']
            
            if len(x_vals) < 2:
                continue
            
            stats += f"\n{series_name}:\n"
            stats += f"  Display Points: {len(x_vals):,}\n"
            
            # Show original points if available
            if series_name in self.original_data:
                orig_len = len(self.original_data[series_name]['x'])
                if orig_len != len(x_vals):
                    reduction = 100 * (1 - len(x_vals) / orig_len)
                    stats += f"  Original Points: {orig_len:,} ({reduction:.1f}% reduction)\n"
            
            stats += f"  X Range: [{min(x_vals):.2f}, {max(x_vals):.2f}]\n"
            stats += f"  Y Range: [{min(y_vals):.2f}, {max(y_vals):.2f}]\n"
            stats += f"  Mean X: {np.mean(x_vals):.2f}\n"
            stats += f"  Mean Y: {np.mean(y_vals):.2f}\n"
            
            if len(x_vals) > 1:
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                stats += f"  Correlation: {correlation:.3f}\n"
            
            stats += "-" * 30 + "\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def toggle_statistics(self):
        """Show or hide statistics panel."""
        if self.show_stats_var.get():
            self.stats_section.grid()
            self.update_statistics()
        else:
            self.stats_section.grid_remove()
    
    def load_csv(self):
        """Load data from CSV file."""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Read CSV file
            df = pd.read_csv(filename)
            
            # Show column selection dialog
            column_dialog = ColumnSelectDialog(self.root, df.columns.tolist())
            
            if column_dialog.result:
                x_col, y_cols = column_dialog.result
                
                # Create series for each selected Y column
                for i, y_col in enumerate(y_cols):
                    series_name = y_col
                    if series_name in self.data_series:
                        series_name = f"{y_col}_{len(self.data_series)}"
                    
                    # Get data
                    x_data = df[x_col].dropna().tolist()
                    y_data = df[y_col].dropna().tolist()
                    
                    # Trim to same length
                    min_len = min(len(x_data), len(y_data))
                    x_data = x_data[:min_len]
                    y_data = y_data[:min_len]
                    
                    if min_len >= 2:
                        # Assign color and style
                        color_idx = len(self.data_series) % len(self.colors)
                        style_idx = len(self.data_series) % len(self.line_styles)
                        
                        # Store in original data
                        self.original_data[series_name] = {
                            'x': x_data.copy(),
                            'y': y_data.copy()
                        }
                        
                        # Initialize display data
                        self.data_series[series_name] = {
                            'x': x_data.copy(),
                            'y': y_data.copy(),
                            'color': self.colors[color_idx],
                            'style': self.line_styles[style_idx],
                            'visible': True
                        }
                        
                        # Apply initial smoothness
                        self.apply_smoothness_to_series(series_name)
                        
                        print(f"Loaded series '{series_name}' with {min_len} points")
                
                self.update_series_list()
                self.update_points_info()
                self.update_graph()
                messagebox.showinfo("Success", f"Loaded {len(y_cols)} series from {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
    
    def update_series_list(self):
        """Update the series checkboxes and combo box."""
        # Clear existing checkboxes
        for widget in self.series_list_frame.winfo_children():
            widget.destroy()
        
        # Create checkboxes for each series
        self.series_vars = {}
        for i, series_name in enumerate(self.data_series.keys()):
            var = tk.BooleanVar(value=self.data_series[series_name]['visible'])
            self.series_vars[series_name] = var
            
            # Create frame for each series
            series_frame = ttk.Frame(self.series_list_frame)
            series_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            
            # Checkbox
            cb = ttk.Checkbutton(series_frame, text=series_name, variable=var,
                               command=lambda name=series_name: self.toggle_series_visibility(name))
            cb.pack(side=tk.LEFT, padx=(0, 10))
            
            # Color indicator
            color = self.data_series[series_name]['color']
            color_canvas = tk.Canvas(series_frame, width=20, height=20, bg=color, 
                                   highlightthickness=1, highlightbackground="black")
            color_canvas.pack(side=tk.LEFT)
            
            # Point count badge
            if series_name in self.original_data:
                orig_count = len(self.original_data[series_name]['x'])
                disp_count = len(self.data_series[series_name]['x'])
                if disp_count < orig_count:
                    badge_text = f"{disp_count}/{orig_count}"
                    badge = ttk.Label(series_frame, text=badge_text, 
                                     font=('Arial', 7), foreground="gray")
                    badge.pack(side=tk.LEFT, padx=(5, 0))
        
        # Update series combo box
        self.series_combo['values'] = list(self.data_series.keys())
        if self.data_series and not self.current_series:
            self.current_series = list(self.data_series.keys())[0]
            self.series_var.set(self.current_series)
            self.update_series_settings()
    
    def toggle_series_visibility(self, series_name):
        """Toggle visibility of a series."""
        if series_name in self.data_series:
            self.data_series[series_name]['visible'] = self.series_vars[series_name].get()
            self.update_graph()
    
    def on_series_selected(self, event):
        """Handle series selection."""
        self.current_series = self.series_var.get()
        self.update_series_settings()
    
    def update_series_settings(self):
        """Update settings for the selected series."""
        if self.current_series and self.current_series in self.data_series:
            series_data = self.data_series[self.current_series]
            self.color_var.set(series_data['color'])
            
            # Convert line style to display format
            style_map = {
                '-': 'Solid (-)',
                '--': 'Dashed (--)',
                '-.': 'Dash-dot (-.)',
                ':': 'Dotted (:)',
                (0, (3, 1, 1, 1)): 'Loosely dotted',
                (0, (5, 5)): 'Dash-dot-dot'
            }
            
            # Reverse lookup
            for key, value in style_map.items():
                if key == series_data['style']:
                    self.style_var.set(value)
                    break
    
    def update_series_color(self, event=None):
        """Update color of selected series."""
        if self.current_series and self.current_series in self.data_series:
            self.data_series[self.current_series]['color'] = self.color_var.get()
            self.update_graph()
            self.update_series_list()  # Update color indicators
    
    def update_series_style(self, event=None):
        """Update line style of selected series."""
        if self.current_series and self.current_series in self.data_series:
            # Convert display format to actual style
            style_map = {
                'Solid (-)': '-',
                'Dashed (--)': '--',
                'Dash-dot (-.)': '-.',
                'Dotted (:)': ':',
                'Loosely dotted': (0, (3, 1, 1, 1)),
                'Dash-dot-dot': (0, (5, 5))
            }
            
            display_style = self.style_var.get()
            if display_style in style_map:
                self.data_series[self.current_series]['style'] = style_map[display_style]
                self.update_graph()
    
    def on_method_change(self, event):
        """Handle interpolation method change."""
        self.selected_method = self.method_var.get()
        self.update_graph()
    
    def add_point(self):
        """Add a point to the selected series."""
        if not self.current_series:
            messagebox.showwarning("No Series", "Please select a series first.")
            return
        
        dialog = PointDialog(self.root, f"Add Point to {self.current_series}", None)
        if dialog.result:
            x, y = dialog.result
            
            # Add to both original and display data
            if self.current_series not in self.original_data:
                self.original_data[self.current_series] = {'x': [], 'y': []}
            
            self.original_data[self.current_series]['x'].append(x)
            self.original_data[self.current_series]['y'].append(y)
            
            # Apply smoothness to update display data
            self.apply_smoothness_to_series(self.current_series)
            
            self.update_points_info()
            self.update_graph()
    
    def edit_point(self):
        """Edit a point in the selected series."""
        if not self.current_series:
            messagebox.showwarning("No Series", "Please select a series first.")
            return
        
        series_data = self.data_series[self.current_series]
        
        # Create point selection dialog
        points_list = [f"({x:.2f}, {y:.2f})" for x, y in zip(series_data['x'], series_data['y'])]
        
        if not points_list:
            messagebox.showwarning("No Points", "The selected series has no points.")
            return
        
        selection_dialog = PointSelectDialog(self.root, "Select Point to Edit", points_list)
        
        if selection_dialog.result is not None:
            index = selection_dialog.result
            old_x, old_y = series_data['x'][index], series_data['y'][index]
            
            dialog = PointDialog(self.root, f"Edit Point {index+1}", (old_x, old_y))
            if dialog.result:
                new_x, new_y = dialog.result
                
                # Find and update in original data (if it exists)
                if self.current_series in self.original_data:
                    # Find the closest point in original data
                    orig_x = self.original_data[self.current_series]['x']
                    orig_y = self.original_data[self.current_series]['y']
                    
                    # Simple approach: update the first matching point
                    # In a real app, you'd want a better mapping
                    if index < len(orig_x):
                        orig_x[index] = new_x
                        orig_y[index] = new_y
                
                # Apply smoothness to update display data
                self.apply_smoothness_to_series(self.current_series)
                
                self.update_points_info()
                self.update_graph()
    
    def delete_point(self):
        """Delete a point from the selected series."""
        if not self.current_series:
            messagebox.showwarning("No Series", "Please select a series first.")
            return
        
        series_data = self.data_series[self.current_series]
        
        # Create point selection dialog
        points_list = [f"({x:.2f}, {y:.2f})" for x, y in zip(series_data['x'], series_data['y'])]
        
        if not points_list:
            messagebox.showwarning("No Points", "The selected series has no points.")
            return
        
        selection_dialog = PointSelectDialog(self.root, "Select Point to Delete", points_list)
        
        if selection_dialog.result is not None:
            index = selection_dialog.result
            
            # Find and delete from original data (if it exists)
            if self.current_series in self.original_data:
                orig_x = self.original_data[self.current_series]['x']
                orig_y = self.original_data[self.current_series]['y']
                
                # Simple approach: delete by index
                # In a real app, you'd want a better mapping
                if index < len(orig_x):
                    del orig_x[index]
                    del orig_y[index]
            
            # Apply smoothness to update display data
            self.apply_smoothness_to_series(self.current_series)
            
            self.update_points_info()
            self.update_graph()
    
    def clear_series(self):
        """Clear all points from the selected series."""
        if not self.current_series:
            messagebox.showwarning("No Series", "Please select a series first.")
            return
        
        if messagebox.askyesno("Clear Series", f"Clear all points from '{self.current_series}'?"):
            if self.current_series in self.original_data:
                self.original_data[self.current_series]['x'].clear()
                self.original_data[self.current_series]['y'].clear()
            
            self.data_series[self.current_series]['x'].clear()
            self.data_series[self.current_series]['y'].clear()
            
            self.update_points_info()
            self.update_graph()
    
    def delete_series(self):
        """Delete the selected series."""
        if not self.current_series:
            messagebox.showwarning("No Series", "Please select a series first.")
            return
        
        if messagebox.askyesno("Delete Series", f"Delete series '{self.current_series}'?"):
            if self.current_series in self.original_data:
                del self.original_data[self.current_series]
            
            del self.data_series[self.current_series]
            self.current_series = None
            self.series_var.set('')
            
            self.update_series_list()
            self.update_points_info()
            self.update_graph()
    
    def clear_all_series(self):
        """Clear all series."""
        if not self.data_series:
            return
        
        if messagebox.askyesno("Clear All", "Delete all series?"):
            self.original_data.clear()
            self.data_series.clear()
            self.current_series = None
            self.series_var.set('')
            
            self.update_series_list()
            self.update_points_info()
            self.update_graph()
    
    def load_sample_data(self):
        """Load sample data with multiple series."""
        # Create noisy data with many points
        np.random.seed(42)
        x_vals = np.linspace(0, 10, 10000)  # 10,000 points!
        
        # Sample data: noisy sine, cosine, and quadratic
        noise_level = 0.2
        
        self.original_data = {
            'Noisy Sine Wave': {
                'x': x_vals.tolist(),
                'y': (np.sin(x_vals) + noise_level * np.random.randn(len(x_vals))).tolist()
            },
            'Noisy Cosine Wave': {
                'x': x_vals.tolist(),
                'y': (np.cos(x_vals) + noise_level * np.random.randn(len(x_vals))).tolist()
            },
            'Noisy Quadratic': {
                'x': x_vals.tolist(),
                'y': (0.1 * x_vals**2 - x_vals + noise_level * np.random.randn(len(x_vals))).tolist()
            }
        }
        
        # Initialize display data
        self.data_series = {}
        for i, (name, data) in enumerate(self.original_data.items()):
            color_idx = i % len(self.colors)
            style_idx = i % len(self.line_styles)
            
            self.data_series[name] = {
                'x': data['x'].copy(),
                'y': data['y'].copy(),
                'color': self.colors[color_idx],
                'style': self.line_styles[style_idx],
                'visible': True
            }
        
        # Apply initial smoothness
        for series_name in self.data_series:
            self.apply_smoothness_to_series(series_name)
        
        # Update labels
        self.plot_title_var.set("Noisy Data with Smoothing")
        self.x_label_var.set("Time (s)")
        self.y_label_var.set("Amplitude")
        self.update_labels()
        
        self.update_series_list()
        self.update_points_info()
        messagebox.showinfo("Sample Data", 
                          f"Loaded noisy sample data with 3 series, each with 10,000 points.\n"
                          f"Use the smoothness slider to reduce data points and smooth the curves.")
    
    def save_image(self):
        """Save the current graph as an image."""
        if not self.data_series:
            messagebox.showwarning("No Data", "No data to save.")
            return
        
        # Get current date and time for default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ask user for file location and format
        filetypes = [
            ("PNG Image", "*.png"),
            ("PDF Document", "*.pdf"),
            ("SVG Vector", "*.svg"),
            ("JPEG Image", "*.jpg;*.jpeg"),
            ("TIFF Image", "*.tiff;*.tif"),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            initialfile=f"graph_smooth{self.smoothness_level}_{timestamp}.png",
            title="Save Graph As"
        )
        
        if filename:
            try:
                # Get DPI from user
                dpi_dialog = DpiDialog(self.root)
                if dpi_dialog.result:
                    dpi = dpi_dialog.result
                else:
                    dpi = 300  # default
                
                # Save the figure
                self.figure.savefig(filename, dpi=dpi, bbox_inches='tight', 
                                  facecolor='white', edgecolor='none')
                
                messagebox.showinfo("Success", f"Graph saved successfully!\n\nFile: {filename}\nDPI: {dpi}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{str(e)}")
    
    def export_data(self):
        """Export all data to CSV file."""
        if not self.data_series:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=f"data_smooth{self.smoothness_level}.csv",
            title="Export Data"
        )
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    headers = ['Series', 'X', 'Y', 'Smoothness_Level']
                    writer.writerow(headers)
                    
                    # Write data
                    for series_name, series_data in self.data_series.items():
                        for x, y in zip(series_data['x'], series_data['y']):
                            writer.writerow([series_name, x, y, self.smoothness_level])
                
                messagebox.showinfo("Export Complete", 
                                  f"Data exported successfully to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def copy_to_csv(self):
        """Copy data to clipboard in CSV format."""
        if not self.data_series:
            messagebox.showwarning("No Data", "No data to copy.")
            return
        
        try:
            csv_data = "Series,X,Y,Smoothness_Level\n"
            for series_name, series_data in self.data_series.items():
                for x, y in zip(series_data['x'], series_data['y']):
                    csv_data += f"{series_name},{x},{y},{self.smoothness_level}\n"
            
            self.root.clipboard_clear()
            self.root.clipboard_append(csv_data)
            
            messagebox.showinfo("Copied", "Data copied to clipboard in CSV format.")
            
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy data:\n{str(e)}")


class ColumnSelectDialog(simpledialog.Dialog):
    def __init__(self, parent, columns):
        self.columns = columns
        self.result = None
        super().__init__(parent, "Select Columns")
    
    def body(self, master):
        ttk.Label(master, text="Select X column:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, pady=(0, 10))
        
        self.x_var = tk.StringVar()
        x_combo = ttk.Combobox(master, textvariable=self.x_var, 
                              values=self.columns, state="readonly", width=20)
        x_combo.grid(row=1, column=0, pady=(0, 20))
        
        ttk.Label(master, text="Select Y columns:", 
                 font=('Arial', 10, 'bold')).grid(row=2, column=0, pady=(0, 10))
        
        # Frame for Y column checkboxes
        self.y_frame = ttk.Frame(master)
        self.y_frame.grid(row=3, column=0, pady=(0, 10))
        
        self.y_vars = {}
        for i, column in enumerate(self.columns):
            var = tk.BooleanVar(value=True if i < 3 else False)
            self.y_vars[column] = var
            
            cb = ttk.Checkbutton(self.y_frame, text=column, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
        
        return x_combo
    
    def validate(self):
        x_col = self.x_var.get()
        if not x_col:
            messagebox.showerror("Error", "Please select an X column.")
            return False
        
        y_cols = [col for col, var in self.y_vars.items() if var.get()]
        if not y_cols:
            messagebox.showerror("Error", "Please select at least one Y column.")
            return False
        
        self.result = (x_col, y_cols)
        return True
    
    def apply(self):
        pass


class PointDialog(simpledialog.Dialog):
    def __init__(self, parent, title, initial_value=None):
        self.initial_value = initial_value
        super().__init__(parent, title)
    
    def body(self, master):
        ttk.Label(master, text="Enter coordinates:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        ttk.Label(master, text="X coordinate:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(master, text="Y coordinate:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        
        if self.initial_value:
            self.x_var.set(str(self.initial_value[0]))
            self.y_var.set(str(self.initial_value[1]))
        
        self.x_entry = ttk.Entry(master, textvariable=self.x_var, width=25)
        self.y_entry = ttk.Entry(master, textvariable=self.y_var, width=25)
        
        self.x_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_entry.grid(row=2, column=1, padx=5, pady=5)
        
        return self.x_entry
    
    def validate(self):
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            self.result = (x, y)
            return True
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for coordinates.")
            return False
    
    def apply(self):
        pass


class PointSelectDialog(simpledialog.Dialog):
    def __init__(self, parent, title, points_list):
        self.points_list = points_list
        self.result = None
        super().__init__(parent, title)
    
    def body(self, master):
        ttk.Label(master, text="Select a point:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, pady=(0, 10))
        
        self.listbox = tk.Listbox(master, height=10, width=30)
        scrollbar = ttk.Scrollbar(master, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        for i, point in enumerate(self.points_list):
            self.listbox.insert(tk.END, f"{i+1}. {point}")
        
        self.listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)
        
        return self.listbox
    
    def validate(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showerror("No Selection", "Please select a point.")
            return False
        
        self.result = selection[0]
        return True
    
    def apply(self):
        pass


class DpiDialog(simpledialog.Dialog):
    def __init__(self, parent):
        self.result = None
        super().__init__(parent, "Image Quality")
    
    def body(self, master):
        ttk.Label(master, text="Select image resolution (DPI):", 
                 font=('Arial', 10)).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        self.dpi_var = tk.StringVar(value="300")
        
        # DPI options
        dpi_options = [
            ("Low (150 DPI)", "150"),
            ("Medium (300 DPI)", "300"),
            ("High (600 DPI)", "600"),
            ("Very High (1200 DPI)", "1200"),
            ("Custom", "custom")
        ]
        
        for i, (text, value) in enumerate(dpi_options):
            ttk.Radiobutton(master, text=text, variable=self.dpi_var, 
                          value=value).grid(row=i+1, column=0, columnspan=2, sticky=tk.W, padx=20)
        
        # Custom DPI entry
        ttk.Label(master, text="Custom DPI:").grid(row=6, column=0, pady=(10, 5), sticky=tk.W)
        self.custom_dpi_var = tk.StringVar()
        self.custom_entry = ttk.Entry(master, textvariable=self.custom_dpi_var, width=10)
        self.custom_entry.grid(row=6, column=1, pady=(10, 5), sticky=tk.W)
        
        return self.custom_entry
    
    def validate(self):
        if self.dpi_var.get() == "custom":
            try:
                dpi = int(self.custom_dpi_var.get())
                if 50 <= dpi <= 2400:
                    self.result = dpi
                    return True
                else:
                    messagebox.showerror("Invalid DPI", "DPI must be between 50 and 2400.")
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for DPI.")
                return False
        else:
            self.result = int(self.dpi_var.get())
            return True
    
    def apply(self):
        pass


def main():
    root = tk.Tk()
    app = GraphInterpolator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
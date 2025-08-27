#!/usr/bin/env python3
"""
Interactive Point Cloud Visualizer for Zivid data
Visualizes point cloud data produced by arm_zivid_ros_node_local.py

Usage:
    python viz_zivid_pc.py -f /path/to/dataset.zarr.zip -i 0
    python viz_zivid_pc.py -f /path/to/dataset.h5 -i 5
"""

import argparse
import numpy as np
import zarr
import h5py
import os
import sys
from pathlib import Path
import glob
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib.util

# Add the parent directory to path to import codecs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# Register custom codecs for zarr files
register_codecs()

class MatplotlibPointCloudVisualizer:
    def __init__(self, file_path, frame_idx=0):
        self.file_path = file_path
        self.current_frame = frame_idx
        self.data_format = self.detect_format()
        
        # Load data
        self.rgb_data, self.depth_data, self.pc_data, self.timestamps = self.load_data()
        self.num_frames = len(self.rgb_data)
        
        print(f"Loaded {self.num_frames} frames from {self.file_path}")
        print(f"Data format: {self.data_format}")
        print(f"RGB shape: {self.rgb_data.shape}")
        print(f"Depth shape: {self.depth_data.shape}")
        print(f"Point cloud shape: {self.pc_data.shape}")
        
        # Initialize matplotlib
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Hover annotation
        self.annotation = None
        self.setup_hover_callback()
        
    def detect_format(self):
        """Detect if the file is zarr, zarr.zip, or h5 format"""
        file_path = str(self.file_path)
        if file_path.endswith('.zarr.zip') or file_path.endswith('.zarr'):
            return 'zarr'
        elif file_path.endswith('.h5'):
            return 'h5'
        else:
            # Try to detect based on content
            try:
                zarr.open(file_path, mode='r')
                return 'zarr'
            except Exception:
                try:
                    h5py.File(file_path, 'r')
                    return 'h5'
                except Exception:
                    raise ValueError(f"Unable to detect format for {file_path}")
    
    def load_data(self):
        """Load data from zarr or h5 file"""
        if self.data_format == 'zarr':
            return self.load_zarr_data()
        elif self.data_format == 'h5':
            return self.load_h5_data()
        else:
            raise ValueError(f"Unsupported format: {self.data_format}")
    
    def load_zarr_data(self):
        """Load data from zarr file"""
        data = zarr.open(self.file_path, mode='r')
        
        rgb = np.array(data['rgb'])
        depth = np.array(data['depth'])
        pc = np.array(data['pc'])
        
        # Load timestamps if available
        timestamps = None
        if 'timestamps' in data:
            timestamps = np.array(data['timestamps'])
            
        return rgb, depth, pc, timestamps
    
    def load_h5_data(self):
        """Load data from h5 file"""
        with h5py.File(self.file_path, 'r') as f:
            rgb = np.array(f['rgb'])
            depth = np.array(f['depth'])
            pc = np.array(f['pc'])
            
            # Load timestamps if available
            timestamps = None
            if 'timestamps' in f:
                timestamps = np.array(f['timestamps'])
                
        return rgb, depth, pc, timestamps
    
    def extract_point_cloud(self, frame_idx):
        """Extract point cloud for a specific frame"""
        if frame_idx >= self.num_frames or frame_idx < 0:
            print(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
            return None, None
        
        pc_frame = self.pc_data[frame_idx]  # Shape: [6, max_points]
        
        # Extract XYZ and RGB
        xyz = pc_frame[:3, :].T  # Shape: [max_points, 3]
        rgb = pc_frame[3:6, :].T  # Shape: [max_points, 3]
        
        # Filter out invalid points (points with inf values)
        valid_mask = np.all(np.isfinite(xyz), axis=1)
        xyz_filtered = xyz[valid_mask]
        rgb_filtered = rgb[valid_mask]
        
        # Normalize RGB to [0, 1] range if needed
        if rgb_filtered.max() > 1.0:
            rgb_filtered = rgb_filtered / 255.0
        
        return xyz_filtered, rgb_filtered
    
    def setup_hover_callback(self):
        """Setup mouse hover callback for coordinate display"""
        def on_hover(event):
            if event.inaxes != self.ax:
                return
            
            if hasattr(self, 'scatter') and self.scatter:
                # Get mouse position in data coordinates
                if event.xdata is None or event.ydata is None:
                    return
                
                # Find closest point to mouse cursor
                xyz, _ = self.extract_point_cloud(self.current_frame)
                if xyz is None or len(xyz) == 0:
                    return
                
                # Project 3D points to 2D screen coordinates
                proj_points = self.ax.transData.transform(xyz)
                mouse_point = np.array([event.x, event.y])
                
                # Calculate distances to all projected points
                distances = np.sqrt(np.sum((proj_points - mouse_point)**2, axis=1))
                
                # Find closest point within threshold
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                # Threshold for considering a point "hovered over"
                hover_threshold = 20  # pixels
                
                if min_distance < hover_threshold:
                    coord = xyz[min_idx]
                    
                    # Remove previous annotation
                    if self.annotation:
                        self.annotation.remove()
                    
                    # Create new annotation
                    annotation_text = f'Point {min_idx}\nX: {coord[0]:.4f}\nY: {coord[1]:.4f}\nZ: {coord[2]:.4f}'
                    self.annotation = self.fig.text(0.02, 0.98, annotation_text,
                                                   fontsize=10,
                                                   verticalalignment='top',
                                                   bbox=dict(boxstyle='round,pad=0.3', 
                                                           facecolor='yellow', 
                                                           alpha=0.8))
                    self.fig.canvas.draw_idle()
                else:
                    # Remove annotation if not hovering over any point
                    if self.annotation:
                        self.annotation.remove()
                        self.annotation = None
                        self.fig.canvas.draw_idle()
        
        self.fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def setup_key_callbacks(self):
        """Setup keyboard callbacks for frame navigation"""
        def on_key_press(event):
            if event.key == 'right' or event.key == 'd':
                self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
                self.update_visualization()
            elif event.key == 'left' or event.key == 'a':
                self.current_frame = max(self.current_frame - 1, 0)
                self.update_visualization()
            elif event.key == 'pageup' or event.key == 'w':
                self.current_frame = min(self.current_frame + 10, self.num_frames - 1)
                self.update_visualization()
            elif event.key == 'pagedown' or event.key == 's':
                self.current_frame = max(self.current_frame - 10, 0)
                self.update_visualization()
            elif event.key == 'home' or event.key == 'q':
                self.current_frame = 0
                self.update_visualization()
            elif event.key == 'end' or event.key == 'e':
                self.current_frame = self.num_frames - 1
                self.update_visualization()
            elif event.key == 'h':
                self.print_help()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    def print_help(self):
        """Print help information"""
        print("\n=== Matplotlib Point Cloud Visualizer Controls ===")
        print("Right Arrow / D: Next frame")
        print("Left Arrow / A: Previous frame")
        print("Page Up / W: Jump forward 10 frames")
        print("Page Down / S: Jump backward 10 frames")
        print("Home / Q: First frame")
        print("End / E: Last frame")
        print("H: Show this help")
        print("Mouse: Hover over points to see coordinates")
        print("Mouse: Left-click drag to rotate, right-click drag to zoom")
        print(f"Current frame: {self.current_frame}/{self.num_frames-1}")
        print("=" * 60)
    
    def update_visualization(self):
        """Update the visualization with the current frame"""
        xyz, rgb = self.extract_point_cloud(self.current_frame)
        
        if xyz is None:
            return
        
        # Clear the axes
        self.ax.clear()
        
        # Create scatter plot
        self.scatter = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                                     c=rgb, s=1.0, alpha=0.6)
        
        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        # Set Z label (3D axes support)
        try:
            self.ax.set_zlabel('Z (m)')
        except AttributeError:
            # Fallback for different matplotlib versions
            pass
        
        timestamp_str = ""
        if self.timestamps is not None:
            timestamp_str = f" | Timestamp: {self.timestamps[self.current_frame]:.3f}s"
        
        title = f'Frame {self.current_frame}/{self.num_frames-1} | {len(xyz)} points{timestamp_str}'
        self.ax.set_title(title)
        
        # Set equal aspect ratio
        # max_range = np.array([xyz[:,0].max()-xyz[:,0].min(), 
        #                      xyz[:,1].max()-xyz[:,1].min(),
        #                      xyz[:,2].max()-xyz[:,2].min()]).max() / 2.0
        # mid_x = (xyz[:,0].max()+xyz[:,0].min()) * 0.5
        # mid_y = (xyz[:,1].max()+xyz[:,1].min()) * 0.5
        # mid_z = (xyz[:,2].max()+xyz[:,2].min()) * 0.5
        # self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # # Set Z limits (3D axes support)
        # try:
        #     self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # except AttributeError:
        #     # Fallback for different matplotlib versions
        #     pass

        # Set hard-coded axis ranges
        self.ax.set_xlim(-0.3, 0.1)
        self.ax.set_ylim(-0.2, 0.1)
        # Set Z limits (3D axes support)
        try:
            self.ax.set_zlim(0.6, 0.9)
        except AttributeError:
            # Fallback for different matplotlib versions
            pass

        # Draw the canvas
        self.fig.canvas.draw()
        
        print(f"Displaying frame {self.current_frame}: {len(xyz)} points{timestamp_str}")
        print("Hover over points to see coordinates. Press 'H' for help.")
    
    def run(self):
        """Run the interactive visualizer"""
        # Set up key callbacks
        self.setup_key_callbacks()
        
        # Set initial visualization
        self.update_visualization()
        
        # Print initial help
        print("\n=== Matplotlib Point Cloud Visualizer ===")
        print("Press 'H' for help with controls")
        print("Hover over points to see their coordinates")
        print(f"Loaded {self.num_frames} frames from {Path(self.file_path).name}")
        print("=" * 50)
        
        # Show the plot
        plt.show()


class ZividPointCloudVisualizer:
    def __init__(self, file_path, frame_idx=0):
        # Import Open3D when needed
        try:
            import open3d as o3d
            self.o3d = o3d
        except ImportError:
            raise ImportError("Open3D is required for this visualizer. Install with: pip install open3d")
        
        self.file_path = file_path
        self.current_frame = frame_idx
        self.data_format = self.detect_format()
        
        # Load data
        self.rgb_data, self.depth_data, self.pc_data, self.timestamps = self.load_data()
        self.num_frames = len(self.rgb_data)
        
        print(f"Loaded {self.num_frames} frames from {self.file_path}")
        print(f"Data format: {self.data_format}")
        print(f"RGB shape: {self.rgb_data.shape}")
        print(f"Depth shape: {self.depth_data.shape}")
        print(f"Point cloud shape: {self.pc_data.shape}")
        
        # Initialize Open3D visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self.point_cloud = self.o3d.geometry.PointCloud()
        
        # Mouse interaction state
        self.mouse_x = 0
        self.mouse_y = 0
        self.show_coordinates = False
        self.hovered_point_idx = None
        self.hovered_coordinates = None
        self.current_xyz = None  # Store current frame's xyz data
        
        # Set up key callbacks
        self.setup_key_callbacks()
        self.setup_coordinate_display()
        
    def detect_format(self):
        """Detect if the file is zarr, zarr.zip, or h5 format"""
        file_path = str(self.file_path)
        if file_path.endswith('.zarr.zip') or file_path.endswith('.zarr'):
            return 'zarr'
        elif file_path.endswith('.h5'):
            return 'h5'
        else:
            # Try to detect based on content
            try:
                zarr.open(file_path, mode='r')
                return 'zarr'
            except Exception:
                try:
                    h5py.File(file_path, 'r')
                    return 'h5'
                except Exception:
                    raise ValueError(f"Unable to detect format for {file_path}")
    
    def load_data(self):
        """Load data from zarr or h5 file"""
        if self.data_format == 'zarr':
            return self.load_zarr_data()
        elif self.data_format == 'h5':
            return self.load_h5_data()
        else:
            raise ValueError(f"Unsupported format: {self.data_format}")
    
    def load_zarr_data(self):
        """Load data from zarr file"""
        data = zarr.open(self.file_path, mode='r')
        
        rgb = np.array(data['rgb'])
        depth = np.array(data['depth'])
        pc = np.array(data['pc'])
        
        # Load timestamps if available
        timestamps = None
        if 'timestamps' in data:
            timestamps = np.array(data['timestamps'])
            
        return rgb, depth, pc, timestamps
    
    def load_h5_data(self):
        """Load data from h5 file"""
        with h5py.File(self.file_path, 'r') as f:
            rgb = np.array(f['rgb'])
            depth = np.array(f['depth'])
            pc = np.array(f['pc'])
            
            # Load timestamps if available
            timestamps = None
            if 'timestamps' in f:
                timestamps = np.array(f['timestamps'])
                
        return rgb, depth, pc, timestamps
    
    def load_multiple_chunks(self, dataset_dir):
        """Load data from multiple chunk files in a directory"""
        # Find all chunk files
        if self.data_format == 'zarr':
            pattern = os.path.join(dataset_dir, "processed_chunk_*.zarr")
        else:
            pattern = os.path.join(dataset_dir, "processed_chunk_*.h5")
        
        chunk_files = sorted(glob.glob(pattern), key=self.natural_sort_key)
        
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found matching pattern {pattern}")
        
        rgb_list, depth_list, pc_list, timestamp_list = [], [], [], []
        
        for chunk_file in chunk_files:
            if self.data_format == 'zarr':
                data = zarr.open(chunk_file, mode='r')
                rgb_chunk = np.array(data['rgb'])
                depth_chunk = np.array(data['depth'])
                pc_chunk = np.array(data['pc'])
                timestamps_chunk = np.array(data['timestamps']) if 'timestamps' in data else None
            else:
                with h5py.File(chunk_file, 'r') as f:
                    rgb_chunk = np.array(f['rgb'])
                    depth_chunk = np.array(f['depth'])
                    pc_chunk = np.array(f['pc'])
                    timestamps_chunk = np.array(f['timestamps']) if 'timestamps' in f else None
            
            rgb_list.append(rgb_chunk)
            depth_list.append(depth_chunk)
            pc_list.append(pc_chunk)
            if timestamps_chunk is not None:
                timestamp_list.append(timestamps_chunk)
        
        # Concatenate all chunks
        rgb_data = np.concatenate(rgb_list, axis=0)
        depth_data = np.concatenate(depth_list, axis=0)
        pc_data = np.concatenate(pc_list, axis=0)
        timestamps_data = np.concatenate(timestamp_list, axis=0) if timestamp_list else None
        
        return rgb_data, depth_data, pc_data, timestamps_data
    
    @staticmethod
    def natural_sort_key(filename):
        """Extract chunk number for natural sorting"""
        match = re.search(r'chunk_(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            return filename
    
    def extract_point_cloud(self, frame_idx):
        """Extract point cloud for a specific frame"""
        if frame_idx >= self.num_frames or frame_idx < 0:
            print(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
            return None, None
        
        pc_frame = self.pc_data[frame_idx]  # Shape: [6, max_points]
        
        # Extract XYZ and RGB
        xyz = pc_frame[:3, :].T  # Shape: [max_points, 3]
        rgb = pc_frame[3:6, :].T  # Shape: [max_points, 3]
        
        # Filter out invalid points (points with inf values)
        valid_mask = np.all(np.isfinite(xyz), axis=1)
        xyz_filtered = xyz[valid_mask]
        rgb_filtered = rgb[valid_mask]
        
        # Normalize RGB to [0, 1] range if needed
        if rgb_filtered.max() > 1.0:
            rgb_filtered = rgb_filtered / 255.0
        
        return xyz_filtered, rgb_filtered
    
    def find_closest_point_simple(self, ray_origin, ray_direction):
        """Find the closest point using simple ray-point distance calculation"""
        if self.current_xyz is None or len(self.current_xyz) == 0:
            return None, None
        
        points = self.current_xyz
        
        # Calculate distance from each point to the ray
        # Using point-to-line distance formula
        distances = []
        for i, point in enumerate(points):
            # Vector from ray origin to point
            to_point = point - ray_origin
            
            # Project onto ray direction
            projection_length = np.dot(to_point, ray_direction)
            
            # Skip points behind the camera
            if projection_length < 0:
                distances.append(float('inf'))
                continue
                
            # Find closest point on ray
            closest_on_ray = ray_origin + projection_length * ray_direction
            
            # Distance from point to ray
            distance = np.linalg.norm(point - closest_on_ray)
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Find the closest point
        if len(distances) == 0 or np.all(distances == float('inf')):
            return None, None
        
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Only return if reasonably close (adjust threshold as needed)
        if min_distance < 0.05:  # 5cm threshold
            return min_idx, points[min_idx]
        
        return None, None
    
    def setup_coordinate_display(self):
        """Setup coordinate display using a more robust approach"""
        def show_point_info(vis):
            """Show information about points in the current view"""
            print("\n=== Point Coordinate Information ===")
            
            if self.current_xyz is not None and len(self.current_xyz) > 0:
                print(f"Total points in current frame: {len(self.current_xyz)}")
                
                # Calculate and show statistics
                xyz_data = self.current_xyz
                x_coords = xyz_data[:, 0]
                y_coords = xyz_data[:, 1]
                z_coords = xyz_data[:, 2]
                
                print(f"X range: {x_coords.min():.4f} to {x_coords.max():.4f}")
                print(f"Y range: {y_coords.min():.4f} to {y_coords.max():.4f}")
                print(f"Z range: {z_coords.min():.4f} to {z_coords.max():.4f}")
                
                print("\nSample coordinates:")
                # Show coordinates of some sample points
                indices = np.linspace(0, len(xyz_data)-1, min(10, len(xyz_data)), dtype=int)
                for i, idx in enumerate(indices):
                    coord = xyz_data[idx]
                    print(f"Point [{idx}]: X={coord[0]:.4f}, Y={coord[1]:.4f}, Z={coord[2]:.4f}")
                
                print("\nTip: Use Open3D's built-in selection tools:")
                print("- Hold Ctrl and click to select points")
                print("- Use mouse wheel to zoom")
                print("- Right-click and drag to rotate view")
            else:
                print("No point cloud data available")
            
            print("=" * 40)
            return False
        
        # Register callback for 'C' key to show coordinate information
        self.vis.register_key_callback(67, show_point_info)  # C key
        
        def enable_selection_mode(vis):
            """Enable point selection mode for detailed coordinate viewing"""
            print("\n=== Point Selection Mode ===")
            print("Instructions:")
            print("1. Hold Ctrl and click on points to select them")
            print("2. Selected points will be highlighted")
            print("3. Press 'P' again to show coordinates of selected points")
            print("4. This mode allows for precise point inspection")
            print("=" * 40)
            
            # Note: Open3D's selection is handled internally
            # The user can manually select points and then press P again to see results
            return False
        
        # Register callback for 'P' key for point selection mode
        self.vis.register_key_callback(80, enable_selection_mode)  # P key
    
    def update_visualization(self):
        """Update the visualization with the current frame"""
        xyz, rgb = self.extract_point_cloud(self.current_frame)
        
        if xyz is None:
            return
        
        # Store current xyz data for mouse interaction
        self.current_xyz = xyz
        
        # Clear existing point cloud
        self.point_cloud.clear()
        
        # Set new point cloud data
        self.point_cloud.points = self.o3d.utility.Vector3dVector(xyz)
        self.point_cloud.colors = self.o3d.utility.Vector3dVector(rgb)
        
        # Update visualization
        self.vis.clear_geometries()
        self.vis.add_geometry(self.point_cloud)
        
        # Update window title with frame info
        timestamp_str = ""
        if self.timestamps is not None:
            timestamp_str = f" | Timestamp: {self.timestamps[self.current_frame]:.3f}s"
        
        # Set point size for better visibility
        self.vis.get_render_option().point_size = 2.0
        
        print(f"Displaying frame {self.current_frame}: {len(xyz)} points{timestamp_str}")
        print("Press 'C' for coordinate info or 'P' for selection mode")
    
    def setup_key_callbacks(self):
        """Setup keyboard callbacks for interactive navigation"""
        def next_frame(vis):
            self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
            self.update_visualization()
            return False
        
        def prev_frame(vis):
            self.current_frame = max(self.current_frame - 1, 0)
            self.update_visualization()
            return False
        
        def next_10_frames(vis):
            self.current_frame = min(self.current_frame + 10, self.num_frames - 1)
            self.update_visualization()
            return False
        
        def prev_10_frames(vis):
            self.current_frame = max(self.current_frame - 10, 0)
            self.update_visualization()
            return False
        
        def first_frame(vis):
            self.current_frame = 0
            self.update_visualization()
            return False
        
        def last_frame(vis):
            self.current_frame = self.num_frames - 1
            self.update_visualization()
            return False
        
        def print_help(vis):
            print("\n=== Zivid Point Cloud Visualizer Controls ===")
            print("Right Arrow / D: Next frame")
            print("Left Arrow / A: Previous frame")
            print("Page Up / W: Jump forward 10 frames")
            print("Page Down / S: Jump backward 10 frames")
            print("Home / Q: First frame")
            print("End / E: Last frame")
            print("C: Show point coordinate information")
            print("P: Enable point selection mode")
            print("H: Show this help")
            print("ESC / X: Exit")
            print("Mouse: Ctrl+Click to select points, wheel to zoom, right-drag to rotate")
            print(f"Current frame: {self.current_frame}/{self.num_frames-1}")
            print("=" * 60)
            return False
        
        # Register key callbacks
        self.vis.register_key_callback(262, next_frame)    # Right arrow
        self.vis.register_key_callback(68, next_frame)     # D key
        self.vis.register_key_callback(263, prev_frame)    # Left arrow
        self.vis.register_key_callback(65, prev_frame)     # A key
        self.vis.register_key_callback(266, next_10_frames)  # Page Up
        self.vis.register_key_callback(87, next_10_frames)   # W key
        self.vis.register_key_callback(267, prev_10_frames)  # Page Down
        self.vis.register_key_callback(83, prev_10_frames)   # S key
        self.vis.register_key_callback(268, first_frame)   # Home
        self.vis.register_key_callback(81, first_frame)    # Q key
        self.vis.register_key_callback(269, last_frame)    # End
        self.vis.register_key_callback(69, last_frame)     # E key
        self.vis.register_key_callback(72, print_help)     # H key
    
    def run(self):
        """Run the interactive visualizer"""
        # Initialize the visualizer
        self.vis.create_window(window_name="Zivid Point Cloud Visualizer", width=1200, height=800)
        
        # Set initial visualization
        self.update_visualization()
        
        # Set camera parameters for better view
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        
        # Print initial help
        print("\n=== Zivid Point Cloud Visualizer ===")
        print("Press 'H' for help with controls")
        print("Press 'C' to show coordinate information")
        print("Press 'P' for point selection mode")
        print(f"Loaded {self.num_frames} frames from {Path(self.file_path).name}")
        print("=" * 50)
        
        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()

def find_dataset_files(dataset_path):
    """Find dataset files in a directory or return the single file"""
    if os.path.isfile(dataset_path):
        return [dataset_path]
    
    # Look for common patterns
    patterns = [
        "processed_chunk_*.zarr",
        "processed_chunk_*.h5",
        "*.zarr.zip",
        "*.zarr",
        "*.h5"
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(dataset_path, pattern)))
    
    if not files:
        raise FileNotFoundError(f"No dataset files found in {dataset_path}")
    
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Point Cloud Visualizer for Zivid data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (Open3D mode):
  Right Arrow / D: Next frame
  Left Arrow / A: Previous frame
  Page Up / W: Jump forward 10 frames
  Page Down / S: Jump backward 10 frames
  Home / Q: First frame
  End / E: Last frame
  C: Show coordinate information
  P: Point selection mode (Ctrl+Click to select)
  H: Show help
  ESC: Exit

Controls (Matplotlib mode):
  Right Arrow / D: Next frame
  Left Arrow / A: Previous frame
  Page Up / W: Jump forward 10 frames
  Page Down / S: Jump backward 10 frames
  Home / Q: First frame
  End / E: Last frame
  H: Show help
  Mouse: Hover over points to see coordinates
  Mouse: Left-click drag to rotate, right-click drag to zoom

Examples:
  python viz_zivid_pc.py -f data.zarr.zip -i 0
  python viz_zivid_pc.py -f dataset.h5 -i 10 --matplotlib
  python viz_zivid_pc.py -f /path/to/dataset/dir --mode open3d
        """
    )
    
    parser.add_argument("-f", "--file", required=True,
                       help="Path to dataset file (.zarr.zip, .zarr, .h5) or directory containing chunk files")
    parser.add_argument("-i", "--index", type=int, default=0,
                       help="Starting frame index (default: 0)")
    parser.add_argument("--mode", choices=['open3d', 'matplotlib'], default='open3d',
                       help="Visualization mode: open3d (default) or matplotlib")
    parser.add_argument("--matplotlib", action='store_true',
                       help="Use matplotlib visualization (shortcut for --mode matplotlib)")
    
    args = parser.parse_args()
    
    # Override mode if matplotlib flag is set
    if args.matplotlib:
        args.mode = 'matplotlib'
    
    try:
        if args.mode == 'matplotlib':
            # Check if matplotlib is available
            if (importlib.util.find_spec("matplotlib") is None or 
                importlib.util.find_spec("mpl_toolkits.mplot3d") is None):
                print("Error: matplotlib is not installed. Please install it using:")
                print("  pip install matplotlib")
                return
            
            print("Using matplotlib visualization")
            visualizer = MatplotlibPointCloudVisualizer(args.file, args.index)
            visualizer.run()
        else:  # open3d mode
            # Check if Open3D is installed
            if importlib.util.find_spec("open3d") is None:
                print("Error: Open3D is not installed. Please install it using:")
                print("  pip install open3d")
                return
            
            import open3d as o3d
            print(f"Using Open3D version: {o3d.__version__}")
            visualizer = ZividPointCloudVisualizer(args.file, args.index)
            visualizer.run()
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Interactive Point Cloud Visualizer for processed H5 dataset
Visualizes point cloud data from robotool processed datasets

Usage:
    python viz_ds_pc.py -f /path/to/dataset.h5 -i 0
    python viz_ds_pc.py -f /path/to/dataset.h5 -i 5
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def load_data(file_path):
    """Load data from h5 file"""
    with h5py.File(file_path, 'r') as f:
        try:
            pc_xyz = np.array(f['data/pc_xyz'])  # Shape: [num_frames, 6, max_points]
            pc_rgb = np.array(f['data/pc_rgb'])  # Shape: [num_frames, 6, max_points]
            timestamps = np.array(f['data/timestamp'])
        except:
            pc_xyz = np.array(f['partial_pc'])  # Shape: [num_frames, max_points, 3]
            pc_rgb = pc_xyz
            timestamps = np.arange(len(pc_xyz))

        # Load episode info
        episode_ends = np.array(f['meta/episode_ends'])
        try:
            episode_names_data = f['meta/episode_name']
            episode_names_array = np.array(episode_names_data)
            episode_names = []
            for name in episode_names_array:
                if isinstance(name, bytes):
                    episode_names.append(name.decode('utf-8'))
                else:
                    episode_names.append(str(name))
        except Exception:
            episode_names = [f"Episode {i}" for i in range(len(episode_ends))]

        return pc_xyz, pc_rgb, timestamps, episode_ends, episode_names

class InteractiveDatasetVisualizer:
    def __init__(self, file_path, frame_idx=0):
        self.file_path = file_path
        self.current_frame = frame_idx
        
        # Load data
        self.pc_xyz, self.pc_rgb, self.timestamps, self.episode_ends, self.episode_names = load_data(file_path)
        self.num_frames = len(self.pc_xyz)
        
        print(f"Loaded {self.num_frames} frames from {self.file_path}")
        print(f"Point cloud shape: {self.pc_xyz.shape}")
        print(f"Episodes: {len(self.episode_names)}")
        for i, name in enumerate(self.episode_names):
            start = 0 if i == 0 else self.episode_ends[i-1]
            end = self.episode_ends[i]
            print(f"  {name}: frames {start}-{end-1} ({end-start} frames)")
        
        # Initialize matplotlib
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Hover annotation
        self.annotation = None
        self.setup_hover_callback()
        
    def extract_point_cloud(self, frame_idx):
        """Extract point cloud for a specific frame"""
        if frame_idx >= self.num_frames or frame_idx < 0:
            print(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
            return None, None
     
        # Extract XYZ and RGB for the specific frame
        xyz = self.pc_xyz[frame_idx]  # Shape: [max_points, 3]
        rgb = self.pc_rgb[frame_idx]  # Shape: [max_points, 3]

        # Filter out invalid points (points with inf or nan values)
        valid_mask = np.all(np.isfinite(xyz), axis=1)
        xyz_filtered = xyz[valid_mask]
        rgb_filtered = rgb[valid_mask]
        
        # Normalize RGB to [0, 1] range if needed
        if len(rgb_filtered) > 0:
            if rgb_filtered.max() > 1.0:
                rgb_filtered = rgb_filtered / 255.0
            # Clip to valid range
            rgb_filtered = np.clip(rgb_filtered, 0.0, 1.0)
        
        return xyz_filtered, rgb_filtered

    def get_episode_info(self, frame_idx):
        """Get episode information for a given frame"""
        episode_idx = 0
        for i, end_idx in enumerate(self.episode_ends):
            if frame_idx < end_idx:
                episode_idx = i
                break
        
        start_idx = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        frame_in_episode = frame_idx - start_idx
        
        episode_name = self.episode_names[episode_idx] if episode_idx < len(self.episode_names) else f"Episode {episode_idx}"
        
        return episode_name, frame_in_episode, end_idx - start_idx

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
        print("\n=== Interactive Dataset Point Cloud Visualizer Controls ===")
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
        
        if xyz is None or len(xyz) == 0:
            print(f"No valid points in frame {self.current_frame}")
            return
        
        # Clear the axes
        self.ax.clear()
        
        # Create scatter plot
        self.scatter = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                                     c=rgb, s=1.0, alpha=0.8)
        
        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        try:
            self.ax.set_zlabel('Z (m)')
        except AttributeError:
            pass  # Some matplotlib versions don't have set_zlabel
        
        # Get episode information
        episode_name, frame_in_episode, episode_length = self.get_episode_info(self.current_frame)
        
        timestamp_str = f"Time: {self.timestamps[self.current_frame]:.3f}s"
        episode_str = f"{episode_name} ({frame_in_episode}/{episode_length-1})"
        
        title = f'Frame {self.current_frame}/{self.num_frames-1} | {len(xyz)} points\n{episode_str} | {timestamp_str}'
        self.ax.set_title(title)
        
        # Draw the canvas
        self.fig.canvas.draw()
        
        print(f"\nDisplaying frame {self.current_frame}: {len(xyz)} points")
        print(f"Episode: {episode_str}")
        print(f"Timestamp: {timestamp_str}")
        print("Hover over points to see coordinates. Press 'H' for help.")

    def run(self):
        """Run the interactive visualizer"""
        # Set up key callbacks
        self.setup_key_callbacks()
        
        # Set initial visualization
        self.update_visualization()
        
        # Print initial help
        print("\n=== Interactive Dataset Point Cloud Visualizer ===")
        print("Press 'H' for help with controls")
        print("Hover over points to see their coordinates")
        print(f"Loaded {self.num_frames} frames from {Path(self.file_path).name}")
        print("=" * 50)
        
        # Show the plot
        plt.show()


class Open3DDatasetVisualizer:
    def __init__(self, file_path, frame_idx=0):
        # Import Open3D when needed
        try:
            import open3d as o3d
            self.o3d = o3d
        except ImportError:
            raise ImportError("Open3D is required for this visualizer. Install with: pip install open3d")
        
        self.file_path = file_path
        self.current_frame = frame_idx
        
        # Load data
        self.pc_xyz, self.pc_rgb, self.timestamps, self.episode_ends, self.episode_names = load_data(file_path)
        self.num_frames = len(self.pc_xyz)
        
        print(f"Loaded {self.num_frames} frames from {self.file_path}")
        print(f"Point cloud shape: {self.pc_xyz.shape}")
        print(f"Episodes: {len(self.episode_names)}")
        for i, name in enumerate(self.episode_names):
            start = 0 if i == 0 else self.episode_ends[i-1]
            end = self.episode_ends[i]
            print(f"  {name}: frames {start}-{end-1} ({end-start} frames)")
        
        # Initialize Open3D visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self.point_cloud = self.o3d.geometry.PointCloud()
        
        # Set up key callbacks
        self.setup_key_callbacks()
        
    def extract_point_cloud(self, frame_idx):
        """Extract point cloud for a specific frame"""
        if frame_idx >= self.num_frames or frame_idx < 0:
            print(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
            return None, None
     
        # Extract XYZ and RGB for the specific frame
        xyz = self.pc_xyz[frame_idx]  # Shape: [max_points, 3]
        rgb = self.pc_rgb[frame_idx]  # Shape: [max_points, 3]

        # Filter out invalid points (points with inf or nan values)
        valid_mask = np.all(np.isfinite(xyz), axis=1)
        xyz_filtered = xyz[valid_mask]
        rgb_filtered = rgb[valid_mask]
        
        # Normalize RGB to [0, 1] range if needed
        if len(rgb_filtered) > 0:
            if rgb_filtered.max() > 1.0:
                rgb_filtered = rgb_filtered / 255.0
            # Clip to valid range
            rgb_filtered = np.clip(rgb_filtered, 0.0, 1.0)
        
        return xyz_filtered, rgb_filtered

    def get_episode_info(self, frame_idx):
        """Get episode information for a given frame"""
        episode_idx = 0
        for i, end_idx in enumerate(self.episode_ends):
            if frame_idx < end_idx:
                episode_idx = i
                break
        
        start_idx = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        frame_in_episode = frame_idx - start_idx
        
        episode_name = self.episode_names[episode_idx] if episode_idx < len(self.episode_names) else f"Episode {episode_idx}"
        
        return episode_name, frame_in_episode, end_idx - start_idx

    def update_visualization(self):
        """Update the visualization with the current frame"""
        xyz, rgb = self.extract_point_cloud(self.current_frame)
        
        if xyz is None or len(xyz) == 0:
            print(f"No valid points in frame {self.current_frame}")
            return
        
        # Clear existing point cloud
        self.point_cloud.clear()
        
        # Set new point cloud data
        self.point_cloud.points = self.o3d.utility.Vector3dVector(xyz)
        self.point_cloud.colors = self.o3d.utility.Vector3dVector(rgb)
        
        # Update visualization
        self.vis.clear_geometries()
        self.vis.add_geometry(self.point_cloud)
        
        # Set point size for better visibility
        self.vis.get_render_option().point_size = 2.0
        
        # Get episode information
        episode_name, frame_in_episode, episode_length = self.get_episode_info(self.current_frame)
        
        timestamp_str = f"Time: {self.timestamps[self.current_frame]:.3f}s"
        episode_str = f"{episode_name} ({frame_in_episode}/{episode_length-1})"
        
        print(f"\nDisplaying frame {self.current_frame}: {len(xyz)} points")
        print(f"Episode: {episode_str}")
        print(f"Timestamp: {timestamp_str}")

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
            print("\n=== Dataset Point Cloud Visualizer Controls ===")
            print("Right Arrow / D: Next frame")
            print("Left Arrow / A: Previous frame")
            print("Page Up / W: Jump forward 10 frames")
            print("Page Down / S: Jump backward 10 frames")
            print("Home / Q: First frame")
            print("End / E: Last frame")
            print("H: Show this help")
            print("ESC / X: Exit")
            print("Mouse: Wheel to zoom, right-drag to rotate")
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
        self.vis.create_window(window_name="Dataset Point Cloud Visualizer", width=1200, height=800)
        
        # Set initial visualization
        self.update_visualization()
        
        # Set camera parameters for better view
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        
        # Print initial help
        print("\n=== Dataset Point Cloud Visualizer ===")
        print("Press 'H' for help with controls")
        print(f"Loaded {self.num_frames} frames from {Path(self.file_path).name}")
        print("=" * 50)
        
        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()


def visualize_frame(file_path, frame_idx, mode='matplotlib'):
    """Visualize a single frame using the specified visualizer"""
    if mode == 'open3d':
        visualizer = Open3DDatasetVisualizer(file_path, frame_idx)
    else:
        visualizer = InteractiveDatasetVisualizer(file_path, frame_idx)
    visualizer.run()

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Point Cloud Visualizer for processed H5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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

Controls (Open3D mode):
  Right Arrow / D: Next frame
  Left Arrow / A: Previous frame
  Page Up / W: Jump forward 10 frames
  Page Down / S: Jump backward 10 frames
  Home / Q: First frame
  End / E: Last frame
  H: Show help
  ESC / X: Exit
  Mouse: Wheel to zoom, right-drag to rotate

Examples:
  python viz_ds_pc.py -f dataset.h5 -i 0
  python viz_ds_pc.py -f dataset.h5 -i 100 --mode open3d
  python viz_ds_pc.py -f dataset.h5 -i 100 --open3d
        """
    )
    
    parser.add_argument("-f", "--file", required=True,
                       help="Path to processed dataset H5 file")
    parser.add_argument("-i", "--index", type=int, default=0,
                       help="Starting frame index (default: 0)")
    parser.add_argument("--mode", choices=['matplotlib', 'open3d'], default='matplotlib',
                       help="Visualization mode: matplotlib (default) or open3d")
    parser.add_argument("--open3d", action='store_true',
                       help="Use Open3D visualization (shortcut for --mode open3d)")
    
    args = parser.parse_args()
    
    # Override mode if open3d flag is set
    if args.open3d:
        args.mode = 'open3d'
    
    try:
        # Check if the file exists
        if not Path(args.file).exists():
            print(f"Error: File {args.file} does not exist")
            return
        
        visualize_frame(args.file, args.index, args.mode)
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

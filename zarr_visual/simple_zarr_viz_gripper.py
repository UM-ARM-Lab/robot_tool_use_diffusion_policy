import zarr
import numpy as np
import matplotlib.pyplot as plt

def visualize_zarr_actions(zarr_file_path, visual_key='robot_act'):
    """
    Simple visualization of robot actions from zarr file
    """
    # Load zarr data
    zarr_data = zarr.open(zarr_file_path, mode='r')
    print(zarr_data.tree())

    robot_actions = np.array(zarr_data['data'][visual_key])
    robot_obs = np.array(zarr_data['data']['robot_obs'])
    
    print(f"Robot actions shape: {robot_actions.shape}")
    print(f"Number of timesteps: {robot_actions.shape[0]}")
    print(f"Number of dimensions: {robot_actions.shape[1]}")
    
    # Create time axis
    time_steps = np.arange(robot_actions.shape[0])
    
    # Plot each dimension vs time
    plt.figure(figsize=(12, 8))
    
    for dim in range(4):
        plt.subplot(6, 6, dim + 1)  # 3 rows, 4 columns for up to 12 dimensions
        plt.plot(time_steps, robot_actions[:, 7+dim])
        plt.plot(time_steps,robot_obs[:, 7+dim], 'g--')  # Plot robot_obs in green dashed line
        # plt.plot(235,robot_actions[235, dim], 'ro')  # Highlight the 235th timestep
        plt.title(f'Dimension {dim}')
        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    zarr_file_path = "./data/victor/dataset_2025-07-15_19-03-32.zarr.zip"
    visualize_zarr_actions(zarr_file_path,visual_key='robot_act')

import numpy as np

def generate_local_obstacle_map(data, resolution, size, depth = 0.2, height = 0.5):
    """
    Generate a local obstacle map from a laser scan data object.

    Parameters:
        data (dict): The scan data, expected to contain:
            - "ranges": list of range measurements (floats)
            - "angle_min": starting angle of the scan (radians)
            - "angle_increment": angular difference between successive measurements (radians)
            - "range_min": minimum valid range measurement
            - "range_max": maximum valid range measurement
        resolution (float): The resolution of the obstacle map (meters per cell).
        size (float): The physical size of the square obstacle map (meters).

    Returns:
        numpy.ndarray: A 2D numpy array (grid) with 1s indicating obstacles and 0s elsewhere.
                       The map is centered on the vehicle.
    """
    # Determine the grid size (number of cells per side)
    grid_size = int(np.ceil(size / resolution))
    obstacle_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Determine the center index of the grid (the car's location)
    center = grid_size // 2

    angles = np.arange(len(data["ranges"])) * data["angle_increment"] + data["angle_min"]
    thick = np.arange(0, depth, resolution)
    valid = (data["ranges"] >= data["range_min"]) & (data["ranges"] <= data["range_max"])
    ranges = data["ranges"][:, np.newaxis] + thick[np.newaxis, :]

    xs = np.cos(angles)[:, np.newaxis] * ranges
    ys = np.sin(angles)[:, np.newaxis] * ranges
    cols = center - np.round(xs / resolution).astype(int)
    rows = center - np.round(ys / resolution).astype(int)
    valid = valid[:,np.newaxis] & (rows >= 0) & (rows < grid_size) & (cols >= 0) & (cols < grid_size)
    cols = cols[valid]
    rows = rows[valid]
    obstacle_map[rows, cols] = height

    return obstacle_map
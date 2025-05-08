import numpy as np
# Define camera matrices
K = np.array([
    [381.0546264648438, 0, 316.6095581054688],
    [0, 380.6285705566406, 248.536376953125],
    [0, 0, 1]
])

T = np.array([
    [0.999986115848177, 0.00201034610955854, 0.00487099778244396, -0.059056371],
    [-0.00202387960198555, 0.999994101493743, 0.00277504758125118, 0.00020030836],
    [-0.00486539024472426, -0.00278486736512245, 0.9999842861223, 0.00059907947],
    [0.0, 0.0, 0.0, 1.0]
])


# Initial point in real world
# x_init, y_init = 8.12750714349904, 8.017006685006187

# Global variable to store the offset (computed from the first point)
global_offset = None

def world_to_image(x_start, y_start, x, y, z=0,is_first_point=False, image_width=640, image_height=480):
    global global_offset

    # Normalize coordinates to start from (0,0,0)
    x_norm = x - x_start
    y_norm = y - y_start

    # Scale up coordinates uniformly to make the path bigger
    scale_factor = 10.0  # Increased to make the path larger
    x_scaled = x_norm * scale_factor
    y_scaled = y_norm * scale_factor

    # Map x movement to horizontal deviation (less sideways movement)
    x_cam = x_scaled * 0.5  # Reduced to make u change less than v

    # Map y movement to forward distance (z in camera frame)
    z_cam = 1.0 + y_scaled * 1.0  # Keep depth scaling to emphasize forward movement

    # Fixed height above ground
    y_cam = 0.2

    # Create 3D point in camera frame
    point_3d = np.array([x_cam, y_cam, z_cam])

    # Convert to homogeneous coordinates
    point_homogeneous = np.append(point_3d, 1)

    # Apply extrinsic transformation
    point_rgb = T @ point_homogeneous
    point_rgb = point_rgb[:3]

    # Project to image plane
    point_2d_homogeneous = K @ point_rgb

    # Convert to pixel coordinates
    u = point_2d_homogeneous[0] / point_2d_homogeneous[2]
    v = point_2d_homogeneous[1] / point_2d_homogeneous[2]

    # Additional scaling in the image plane to control u vs v rate of change
    u_scale = 0.8  # Reduce the rate of change of u
    v_scale = 2.5  # Increase the rate of change of v to stretch vertically
    u = u * u_scale
    v = v * v_scale

    # Compute the offset for the first point to start at the bottom center
    if is_first_point:
        # Desired starting position: bottom center of the image
        desired_u = image_width / 2  # e.g., 640 for 1280px width
        desired_v = image_height     # e.g., 720 for 720px height

        # Compute the offset to move this point to the desired position
        u_offset = desired_u - u
        v_offset = desired_v - v
        global_offset = (u_offset, v_offset)

    # Apply the offset to all points (including the first point)
    if global_offset is not None:
        u += global_offset[0]
        v += global_offset[1]

    return (u, v) # Return a tuple (u, v)


def transform_coordinates(x, y):
    # Reset the global offset before processing the path
    global global_offset
    global_offset = None

    # Project the path points to the image plane
    projected_points = []
    for i, (a, b) in enumerate(zip(x, y)):
        z = 0  # Assuming z=0 for all points (2D path)
        uv_tuple = world_to_image(x[0], y[0], a, b, z, is_first_point=(i == 0))
        projected_points.append(uv_tuple)

    return projected_points # Return the list of tuples

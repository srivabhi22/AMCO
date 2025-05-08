import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def process_point_cloud(file_path):
    # Load the Point Cloud
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    # Step 1: Filter for Region of Interest (Very Near Region)
    x_min, x_max = -1.5, 1.5  # 1.5 meters left-right of the robot
    y_min, y_max = -0.5, 0.5  # 0.5 meters above and below robot height
    z_min, z_max = 0.1, 2.0   # Points in front and above ground level
    roi_mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &  # X bounds
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &  # Y bounds
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)    # Z bounds
    )
    filtered_points = points[roi_mask]

    # Step 2: Remove Floor Points using Plane Segmentation (RANSAC)
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    plane_model, inliers = pcd_filtered.segment_plane(
        distance_threshold=0.02,  # Distance threshold for floor points
        ransac_n=3,
        num_iterations=1000
    )
    floor_points = filtered_points[inliers]  # Points identified as floor
    non_floor_points = filtered_points[np.setdiff1d(np.arange(len(filtered_points)), inliers)]

    # Step 3: Voxel Downsampling for Computational Efficiency
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(non_floor_points)
    downsampled_pcd = pcd_downsampled.voxel_down_sample(voxel_size=0.05)
    downsampled_points = np.asarray(downsampled_pcd.points)

    # Step 4: Cluster Remaining Points Using DBSCAN
    obstacle_coordinates = []
    if len(downsampled_points) > 0:
        clustering = DBSCAN(eps=0.08, min_samples=15).fit(downsampled_points)
        labels = clustering.labels_

        # Extract Cluster Centroids
        for label in set(labels):
            if label == -1:
                continue  # Ignore noise points
            cluster_points = downsampled_points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            obstacle_coordinates.append((2*float(centroid[0]), float(2*centroid[1]), float(2*centroid[2])))

    # Visualization Step: Create Point Clouds for Visualization
    original_pcd = pcd  # Original point cloud
    floor_pcd = o3d.geometry.PointCloud()  # Floor points
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.paint_uniform_color([0, 0, 1])  # Blue for floor points

    obstacle_pcd = o3d.geometry.PointCloud()  # Obstacle points
    obstacle_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    obstacle_pcd.paint_uniform_color([0, 0, 0])  # Black for obstacles

    # Add Robot Position
    robot_position = np.array([[0, 0, 0]])  # Robot's position at the origin
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_position)
    robot_pcd.paint_uniform_color([0, 1, 0])  # Green for robot

    # # Step 5: Visualize Filtering Stages
    # print(f"Detected Obstacles: {len(obstacle_coordinates)}")
    # for i, coord in enumerate(obstacle_coordinates):
    #     print(f"Obstacle {i+1}: {coord}")

    return obstacle_coordinates


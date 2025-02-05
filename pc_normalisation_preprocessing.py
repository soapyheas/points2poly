import numpy as np
import open3d as o3d
import sys
import os

def load_ply(filepath):
    """Load a binary PLY file as a NumPy array using Open3D."""
    pcd = o3d.io.read_point_cloud(filepath)  # Open3D auto-detects binary PLY
    return np.asarray(pcd.points), pcd

def load_xyz(filepath):
    """Load an XYZ file as a NumPy array."""
    return np.loadtxt(filepath)

def save_ply(filepath, points, original_pcd):
    """Save a NumPy array as a binary PLY file using Open3D, preserving color and normals if present."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Preserve color and normals if they exist in the original point cloud
    if original_pcd.has_colors():
        pcd.colors = original_pcd.colors
    if original_pcd.has_normals():
        pcd.normals = original_pcd.normals

    o3d.io.write_point_cloud(filepath, pcd, write_ascii=False)  # Save as binary PLY

def center_and_scale(source_points, target_points):
    """Centers the source point cloud at origin and scales it to match the target's max bounding box dimension."""
    
    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    # Center source at origin
    source_points_centered = source_points - source_centroid
    
    # Compute max bounding box dimension for scaling
    source_bbox = np.max(source_points_centered, axis=0) - np.min(source_points_centered, axis=0)
    target_bbox = np.max(target_points - target_centroid, axis=0) - np.min(target_points - target_centroid, axis=0)

    source_max_dim = np.max(source_bbox)
    target_max_dim = np.max(target_bbox)

    scale_factor = target_max_dim / source_max_dim if source_max_dim > 0 else 1

    # Scale the source points
    source_points_transformed = source_points_centered * scale_factor

    return source_points_transformed

def main():
    if len(sys.argv) < 3:
        print("Usage: python pc_normalisation_batch.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]  # Folder containing input .ply files
    output_folder = sys.argv[2]  # Folder to save output .ply files

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the reference point cloud (full_view.xyz)
    full_view_points = load_xyz("normalisation_target.xyz")

    # Process all .ply files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".ply"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            print(f"Processing {input_filepath} -> {output_filepath}")

            # Load the point cloud
            source_points, original_pcd = load_ply(input_filepath)

            # Center & scale
            transformed_points = center_and_scale(source_points, full_view_points)

            # Save output as binary PLY
            save_ply(output_filepath, transformed_points, original_pcd)

    print("Batch normalisation complete.")

if __name__ == "__main__":
    main()

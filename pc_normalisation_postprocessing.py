import numpy as np
import open3d as o3d
import trimesh
import sys
import os

def load_mesh(filepath):
    """Load a mesh (.obj) using Trimesh to preserve original geometry."""
    mesh = trimesh.load_mesh(filepath, process=False)  # Keeps original face structure
    return mesh

def load_ply(filepath):
    """Load a binary PLY file as an Open3D PointCloud."""
    pcd = o3d.io.read_point_cloud(filepath)
    if not pcd.has_points():
        raise ValueError(f"PointCloud at {filepath} is empty or unreadable.")
    return pcd

def save_mesh(filepath, mesh):
    """Save the mesh as an .obj file using Trimesh, preserving original faces and structure."""
    mesh.export(filepath)

def align_translation(input_mesh, reference_pcd):
    """Translate input mesh to have the same center as the reference point cloud."""
    input_vertices = input_mesh.vertices.copy()
    reference_center = np.mean(np.asarray(reference_pcd.points), axis=0)
    input_center = np.mean(input_vertices, axis=0)

    # Compute translation vector
    translation = reference_center - input_center

    # Apply translation
    input_vertices += translation

    # Update mesh with translated vertices
    input_mesh.vertices = input_vertices
    return input_mesh

def scale_to_match(input_mesh, reference_pcd):
    """Scales the input mesh to match the reference point cloud's bounding box size."""
    input_vertices = input_mesh.vertices.copy()

    # Compute bounding box sizes
    input_bounds = np.max(input_vertices, axis=0) - np.min(input_vertices, axis=0)
    reference_bounds = reference_pcd.get_max_bound() - reference_pcd.get_min_bound()

    # Compute scale factor (mean ratio of bounding box sizes)
    scale_factor = np.mean(reference_bounds / input_bounds)

    # Scale the mesh around its center
    input_center = np.mean(input_vertices, axis=0)
    input_vertices = (input_vertices - input_center) * scale_factor + input_center

    # Update mesh with scaled vertices
    input_mesh.vertices = input_vertices
    return input_mesh

def icp_refinement(source_mesh, target_pcd):
    """Refines alignment using ICP while keeping the original mesh structure."""
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_mesh.vertices)

    # Initial transformation matrix
    trans_init = np.identity(4)

    # ICP thresholds (coarse to fine)
    icp_thresholds = [0.5, 0.05, 0.01]

    for threshold in icp_thresholds:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        # Apply ICP transformation to mesh vertices
        source_mesh.vertices = (source_mesh.vertices @ reg_p2p.transformation[:3, :3].T) + reg_p2p.transformation[:3, 3]

    return source_mesh

def process_folder(input_folder, reference_folder, output_folder):
    """Processes all .obj files in the input folder, aligning them to corresponding .ply files in reference folder."""

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            input_filepath = os.path.join(input_folder, filename)
            reference_filepath = os.path.join(reference_folder, filename.replace(".obj", ".ply"))
            output_filepath = os.path.join(output_folder, filename)

            if not os.path.exists(reference_filepath):
                print(f"Warning: Reference PLY file not found for {filename}, skipping.")
                continue

            print(f"Processing {input_filepath} -> {output_filepath}")

            # Load input mesh and reference point cloud
            input_mesh = load_mesh(input_filepath)
            reference_pcd = load_ply(reference_filepath)

            # Align translation first (before scaling)
            input_mesh = align_translation(input_mesh, reference_pcd)

            # Scale to match reference bounding box
            input_mesh = scale_to_match(input_mesh, reference_pcd)

            # Fine-tune with ICP
            refined_mesh = icp_refinement(input_mesh, reference_pcd)

            # Save the final transformed mesh
            save_mesh(output_filepath, refined_mesh)

    print("Post-processing complete.")

def main():
    if len(sys.argv) < 4:
        print("Usage: python pc_post_processing.py <input_folder> <reference_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    reference_folder = sys.argv[2]
    output_folder = sys.argv[3]

    process_folder(input_folder, reference_folder, output_folder)

if __name__ == "__main__":
    main()

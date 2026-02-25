import sys
import numpy as np
import matplotlib.pyplot as plt

def visualize_ply_interactive(ply_path):
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
    except ImportError:
        # Fallback to simple parsing if open3d is missing
        print("Open3D not found, parsing point cloud manually...")
        points = []
        colors = []
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        in_header = True
        for line in lines:
            line = line.strip()
            if line == 'end_header':
                in_header = False
                continue
            if not in_header:
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(parts) >= 6:
                    colors.append([float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0])
        points = np.array(points)
        colors = np.array(colors)
        if len(colors) == 0:
            colors = None

    if len(points) == 0:
        print("No points found.")
        return

    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"Interactive Point Cloud: {ply_path}")
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None and len(colors) == len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Optional: ensure aspect ratio is equal
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Tweak the view angle slightly
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python interact_ply.py <input.ply>")
        sys.exit(1)
    visualize_ply_interactive(sys.argv[1])

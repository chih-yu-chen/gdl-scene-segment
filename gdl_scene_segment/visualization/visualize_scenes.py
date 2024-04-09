import json
import open3d as o3d
from open3d.visualization import draw_geometries, Visualizer
from pathlib import Path



def check_mesh_properties(mesh) -> None:

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    print(f"  edge_manifold:          {edge_manifold}")
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    vertex_manifold = mesh.is_vertex_manifold()
    print(f"  vertex_manifold:        {vertex_manifold}")
    self_intersecting = mesh.is_self_intersecting()
    print(f"  self_intersecting:      {self_intersecting}")
    watertight = mesh.is_watertight()
    print(f"  watertight:             {watertight}")
    orientable = mesh.is_orientable()
    print(f"  orientable:             {orientable}")

    return

def render_to_image(img_path:Path,
                    mesh,
                    trajectory,
                    ) -> None:

    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    ctr.set_front(trajectory["front"])
    ctr.set_lookat(trajectory["lookat"])
    ctr.set_up(trajectory["up"])
    ctr.set_zoom(trajectory["zoom"])

    vis.poll_events()
    vis.update_renderer()

    print(f"saving image at {img_path}")
    vis.capture_screen_image(img_path.as_posix(), False)
    vis.destroy_window()

    return



gt_dir = Path("/media/cychen/HDD/scannet/scans")
scene_img_dir = Path("visualizations/scene_imgs")
scene_img_dir.mkdir(parents=True, exist_ok=True)
gt_out_dir = Path("visualizations/gt_imgs")
gt_out_dir.mkdir(parents=True, exist_ok=True)
pred_dir = Path("visualizations/pred_scenes/03_baseline_noGradientRotation")
out_dir = Path("visualizations/pred_imgs/03_baseline_noGradientRotation")
out_dir.mkdir(parents=True, exist_ok=True)

val_split = "gdl_scene_segment/datasets/splits/scannetv2_val.txt"
with open(val_split, 'r') as f:
    scenes = f.read().splitlines()

with open ("gdl_scene_segment/visualization/views.json", 'r') as f:
    views = json.load(f)



for i, view in enumerate(views):

    trajectory = view['trajectory'][0]
    scene = view['scene']

    scene_path = gt_dir/ scene/ f"{scene}_vh_clean_2.ply"
    scene_mesh = o3d.io.read_triangle_mesh(scene_path.as_posix())
    # draw_geometries([scene_mesh])
    render_to_image(scene_img_dir/ f"{scene}.png", scene_mesh, trajectory)

    gt_path = gt_dir/ scene/ f"{scene}_vh_clean_2.labels.ply"
    gt_mesh = o3d.io.read_triangle_mesh(gt_path.as_posix())
    # draw_geometries([gt_mesh])
    render_to_image(gt_out_dir/ f"{scene}.png", gt_mesh, trajectory)

    pred_path = pred_dir/ f"{scene}.ply"
    pred_mesh = o3d.io.read_triangle_mesh(pred_path.as_posix())
    # draw_geometries([pred_mesh])
    render_to_image(out_dir/ f"{scene}.png", pred_mesh, trajectory)

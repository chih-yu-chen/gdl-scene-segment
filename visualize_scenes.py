import copy
import json
import numpy as np
import open3d as o3d
from open3d.visualization import draw_geometries, Visualizer
from pathlib import Path



def get_views(scene_list):

    for i, scene in enumerate(scene_list):
        print(i, scene)
        gt = f"vis/gts/{scene}_labels.ply"
        # gt = f"/media/cychen/HDD/scannet/scans/{scene}/{scene}_vh_clean_2.ply"
        gt_mesh = o3d.io.read_triangle_mesh(gt)
        draw_geometries([gt_mesh])

    return

def check_mesh_properties(mesh):

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

def render_to_image(filename, mesh, trajectory):

    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(trajectory["field_of_view"])
    ctr.set_front(trajectory["front"])
    ctr.set_lookat(trajectory["lookat"])
    ctr.set_up(trajectory["up"])
    ctr.set_zoom(trajectory["zoom"])

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    print(f"saving image at {filename}")
    vis.capture_screen_image(filename, False)
    vis.destroy_window()

    return

def render_all_images(scene_list, view_list, run_name):

    for i, (scene, view) in enumerate(zip(scene_list, view_list)):

        print(i, scene)
        gt = f"vis/gts/{scene}_labels.ply"
        gt_mesh = o3d.io.read_triangle_mesh(gt)
        pred = f"vis/preds/{run_name}/{scene}_labels.ply"
        pred_mesh = o3d.io.read_triangle_mesh(pred)

        traj = view['trajectory'][0]
        Path(f"vis/imgs/{run_name}").mkdir(parents=True, exist_ok=True)
        render_to_image(f"vis/imgs/{run_name}/{scene}_gt.png", gt_mesh, traj)
        render_to_image(f"vis/imgs/{run_name}/{scene}_pred.png", pred_mesh, traj)

    return



if __name__ == '__main__':

    val_split = "ScanNet/Tasks/Benchmark/scannetv2_val.txt"
    with open(val_split, 'r') as f:
        scene_list = f.read().splitlines()

    i = 20
    gt = f"vis/gts/{scene_list[i]}_labels.ply"
    gt_mesh = o3d.io.read_triangle_mesh(gt)
    draw_geometries([gt_mesh])
    # check_mesh_properties(gt_mesh)

    # get_views(scene_list)

    # with open ("vis/view.json", "r") as f:
    #     view_list = json.load(f)

    # render_all_images(scene_list, view_list, "hal_20_10")

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 7.0689835548400879, 8.4662561416625977, 2.3702223300933838 ],
# 			"boundingbox_min" : [ 0.21197108924388885, 1.7900252342224121, -0.041525058448314667 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.61335846383802073, 0.17813731533866362, 0.76945337202600983 ],
# 			"lookat" : [ 2.8794633696215035, 5.6941831504471239, -0.15466299098821007 ],
# 			"up" : [ 0.77688469841237284, -0.039417528452519598, 0.62840784831462082 ],
# 			"zoom" : 0.45999999999999974
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }
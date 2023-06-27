import json
import open3d as o3d
from open3d.visualization import draw_geometries, Visualizer



def get_views(scene_list):

    for i, scene in enumerate(scene_list):
        print(i, scene)
        gt = f"vis/gts/{scene}_labels.ply"
        gt_mesh = o3d.io.read_triangle_mesh(gt)
        draw_geometries([gt_mesh])

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
        render_to_image(f"vis/imgs/{run_name}/{scene}_gt.png", gt_mesh, traj)
        render_to_image(f"vis/imgs/{run_name}/{scene}_pred.png", pred_mesh, traj)

    return



if __name__ == '__main__':

    val_split = "ScanNet/Tasks/Benchmark/300000/scannetv2_val.txt"
    with open(val_split, 'r') as f:
        scene_list = f.read().splitlines()

    # get_views(scene_list)

    with open ("vis/view.json", "r") as f:
        view_list = json.load(f)

    render_all_images(scene_list, view_list, "hal_50_12")

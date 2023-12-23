git clone https://github.com/chih-yu-chen/gdl-scene-segment.git
cd gdl-scene-segment
git clone https://github.com/nmwsharp/diffusion-net.git
poetry install
sh ./get_compile_vcglib.sh
cd gdl_scene_segment/datasets
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test --remove_disconnection --fill_holes 1e6
python simplify_meshes.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30 --test
python simplify_meshes.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30

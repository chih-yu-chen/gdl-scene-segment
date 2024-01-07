## Set up Environment
```
git clone https://github.com/chih-yu-chen/gdl-scene-segment.git
cd gdl-scene-segment
git clone https://github.com/nmwsharp/diffusion-net.git
poetry install
sh ./get_compile_vcglib.sh
```

## Prepare Dataset
```
cd gdl_scene_segment/datasets
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test --remove_disconnection --fill_holes 1e6
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --remove_disconnection --fill_holes 1e6
python simplify_meshes.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30 --test
python simplify_meshes.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30
python compute_operators_hierarchy.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30
```

## Configure Experiment
```
export NAME="name" # set experiment name
cd ../../experiments
mkdir $NAME
cp ../gdl_scene_segment/config/settings.toml $NAME/experiment.toml
vim $NAME/experiment.toml # configure your experiment
```

## Train
```
cd ..
export SETTINGS_FILES_FOR_DYNACONF="gdl_scene_segment/config/settings.toml;experiments/$NAME/experiment.toml"
python gdl_scene_segment/main_hierarchy.py --data_dir DATA_DIR --gpu GPU
```

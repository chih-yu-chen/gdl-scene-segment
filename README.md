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
# put ScanNet's root at DATA_DIR
cd gdl_scene_segment/datasets
```

### mesh-preprocessing:
- just center
```
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test
python preprocess_meshes.py --data_dir DATA_DIR --dst DST
```
- remove disconnection
```
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test --remove_disconnection
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --remove_disconnection
```
- mesh-preprocessing: both
```
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test --remove_disconnection --fill_holes 1e6
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --remove_disconnection --fill_holes 1e6
```

### mesh simplification
```
python simplify_meshes.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30 --test
python simplify_meshes.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30
```

### pre-compute operators
- for vanilla DiffusionNet
```
python compute_operators.py --data_dir DATA_DIR --preprocess PREPROCESS
python compute_operators_hierarchy.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30
```
- for the proposed architecture
```
python compute_operators_hierarchy.py --data_dir DATA_DIR --preprocess PREPROCESS --level_params 0.02 30 30 30
```


## Configure Experiment

```
export NAME="name" # set experiment name
cd ../../experiments
mkdir $NAME
cp ../gdl_scene_segment/config/settings.toml $NAME/settings.toml
vim $NAME/settings.toml # configure your experiment
```

## Train

```
cd ..
export SETTINGS_FILES_FOR_DYNACONF="experiments/$NAME/settings.toml"
```
- train vanilla DiffusionNet
```
python gdl_scene_segment/main.py --data_dir DATA_DIR --gpu GPU
```
- train the proposed architecture
```
python gdl_scene_segment/main_hierarchy.py --data_dir DATA_DIR --gpu GPU
```
- train the proposed architecture with 2 levels and only geodesic branch
```
python gdl_scene_segment/main_hierarchy_geo.py --data_dir DATA_DIR --gpu GPU
```

## Set up Environment

```
git clone https://github.com/chih-yu-chen/gdl-scene-segment.git
cd gdl-scene-segment
git clone https://github.com/nmwsharp/diffusion-net.git
poetry install
sh ./get_compile_vcglib.sh
```


## Prepare Dataset
- First, download the ScanNet dataset to `DATA_DIR`
- Then:
```
cd gdl_scene_segment/datasets
```

### Mesh Pre-processing:
- Preprocess the meshes and save to `DST`
- Center meshes
```
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test
python preprocess_meshes.py --data_dir DATA_DIR --dst DST
```
- Remove disconnection
```
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test --remove_disconnection
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --remove_disconnection
```
- Both mesh pre-processes
```
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --test --remove_disconnection --fill_holes 1e6
python preprocess_meshes.py --data_dir DATA_DIR --dst DST --remove_disconnection --fill_holes 1e6
```

### Mesh Simplification
```
python simplify_meshes.py --data_dir DATA_DIR --preprocess DST --level_params 0.02 30 30 30 --test
python simplify_meshes.py --data_dir DATA_DIR --preprocess DST --level_params 0.02 30 30 30
```

### Pre-compute Operators
- For vanilla DiffusionNet
```
python compute_operators.py --data_dir DATA_DIR --preprocess DST
```
- For the proposed architecture
```
python compute_operators.py --data_dir DATA_DIR --preprocess DST --n_levels 4
```


## Configure Experiment
- Set experiment name
```
export NAME="your_exp_name"
cd ../../experiments
```
- Make experiment folder and configurations
```
mkdir $NAME
cp ../gdl_scene_segment/config/settings.toml $NAME/settings.toml
```
- Configure your experiment
```
vim $NAME/settings.toml
```

## Train

- Let dynaconf know which set of settings is used
```
cd ..
export SETTINGS_FILES_FOR_DYNACONF="experiments/$NAME/settings.toml"
```
- Train vanilla DiffusionNet
```
python gdl_scene_segment/main.py --data_dir DATA_DIR --gpu GPU
```
- Train the proposed architecture
```
python gdl_scene_segment/main_hierarchy.py --data_dir DATA_DIR --gpu GPU
```
- Train the proposed architecture with 2 levels and only geodesic branch
```
python gdl_scene_segment/main_hierarchy_geo.py --data_dir DATA_DIR --gpu GPU
```

## Evaluate
- Download the experiment settings, model checkpoints, and metrics [HERE](https://drive.google.com/file/d/1qbKlkxRBRVvuOToaJhtHXiE9OEvK2m7f/view?usp=sharing)
- Note that some experiments involve structural changes to the architecture, so they are not reproducible
- These experiment folders do not contain `settings.toml`
- Let dynaconf know which set of settings is used
```
export SETTINGS_FILES_FOR_DYNACONF="experiments/$NAME/settings.toml"
```
- Evaluate vanilla DiffusionNet
```
python gdl_scene_segment/main.py --data_dir DATA_DIR --gpu GPU --evaluate
```
- Evaluate the proposed architecture
```
python gdl_scene_segment/main_hierarchy.py --data_dir DATA_DIR --gpu GPU --evaluate
```
- Evaluate the proposed architecture with 2 levels and only geodesic branch
```
python gdl_scene_segment/main_hierarchy_geo.py --data_dir DATA_DIR --gpu GPU --evaluate
```

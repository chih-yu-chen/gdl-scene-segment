#!/bin/bash
PATH_SPLIT="gdl_scene_segment/datasets/splits/scannetv2_val.txt"
RUN_NAME="03_baseline_noGradientRotation"
mkdir "visualizations/pred_scenes/"$RUN_NAME

readarray -t VALSET < $PATH_SPLIT

for i in "${VALSET[@]}"; do

	# echo "generating ground truth visualization $i"
	# PATH_GT="/media/cychen/HDD/scannet/gts/"$i"_labels.txt"
	# PATH_MESH="/media/cychen/HDD/scannet/scans/"$i"/"$i"_vh_clean_2.ply"
	# PATH_OUT="vis/gts/"$i"_labels.ply"
	# python2 ScanNet/BenchmarkScripts/3d_helpers/visualize_labels_on_mesh.py --pred_file $PATH_GT --mesh_file $PATH_MESH --output_file $PATH_OUT

	echo "generating prediction visualization $i"
	PATH_PRED="experiments/baseline/"$RUN_NAME"/preds/"$i".txt"
	PATH_MESH="/media/cychen/HDD/scannet/scans/"$i"/"$i"_vh_clean_2.ply"
	PATH_OUT="visualizations/pred_scenes/"$RUN_NAME"/"$i".ply"
	python3 gdl_scene_segment/visualization/visualize_labels_on_mesh.py --pred_file $PATH_PRED --mesh_file $PATH_MESH --output_file $PATH_OUT
	
done

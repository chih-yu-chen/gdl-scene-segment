readarray -t valset < ScanNet/Tasks/Benchmark/300000/scannetv2_val.txt

for i in "${valset[@]}"; do
	echo "copying ground truth $i"
	cp /media/cychen/HDD/scannet/scans/"$i"/"$i"_vh_clean_2.labels.ply vis/gts/
	echo "generating prediction visualization $i"
	python2 ScanNet/BenchmarkScripts/3d_helpers/visualize_labels_on_mesh.py --pred_file /media/cychen/HDD/scannet/predictions/"$i"_labels.txt --mesh_file /media/cychen/HDD/scannet/scans/"$i"/"$i"_vh_clean_2.ply --output_file vis/predictions/"$i"_labels.ply
done

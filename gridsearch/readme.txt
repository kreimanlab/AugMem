#dataset, GPU id 0, GPU id 1
./toybox_setup_grid_raw.sh toybox 1 3
./toybox_gridsearch_raw.sh
cp -r  toybox_gridsearch_outputs/ ../
cp -r gridsearches ../scripts/
chmod -R +x ../scripts
mv ../scripts/gridsearches ../scripts/gridsearches_toybox
manually remove "--validate" from iCARL
./scripts/combined_gridsearch_toybox.sh
./scripts/combined_gridsearch_toybox_gpu2.sh

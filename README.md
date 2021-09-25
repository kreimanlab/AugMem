# Hypothesis-driven Stream Learning with Augmented Memory
Authors: Mengmi Zhang*, Rohil Badkundri*, Morgan Talbot, Rushikesh Zawar, Gabriel Kreiman (* equal contribution)

Manuscript download [HERE](http://arxiv.org/abs/2104.02206); supplementary download [HERE](https://d2b38104-6cb6-430b-95b9-765197711bda.usrfiles.com/ugd/d2b381_d96d5c9ef91642afa421b9ef53cfb6dc.pdf)

## Project description 

Stream learning refers to the ability to acquire and transfer knowledge across a continuous stream of data without forgetting and without repeated passes over the data. A common way to avoid catastrophic forgetting is to intersperse new examples with replays of old examples stored as image pixels or reproduced by generative models. Here, we considered stream learning in image classification tasks and proposed a novel hypotheses-driven Augmented Memory Network, which efficiently consolidates previous knowledge with a limited number of hypotheses in the augmented memory and replays relevant hypotheses to avoid catastrophic forgetting. The advantages of hypothesis-driven replay over image pixel replay and generative replay are two-fold. First, hypothesis-based knowledge consolidation avoids redundant information in the image pixel space and makes memory usage more efficient. Second, hypotheses in the augmented memory can be re-used for learning new tasks, improving generalization and transfer learning ability. We evaluated our method on three stream learning object recognition datasets. Our method performs comparably well or better than SOTA methods, while offering more efficient memory usage.

## Setup

This PYTORCH project was developed and tested using Ubuntu version 20.04, CUDA 10.1, and Python version 3.6. See ```requirements.txt``` for package versions. Additional requirements: ffmpeg

Refer to [link](https://www.anaconda.com/distribution/) for Anaconda installation. Alternatively, execute the following command:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
```
After Anaconda installation, create a conda environment (here, our conda environment is called "augmem"):
```
conda create -n augmem python=3.6
```
Activate the conda environment:
```
conda activate augmem
```
In the conda environment, 
```
pip install -r requirements.txt
```
Install ffmpeg (the final command is to verify installation):
```
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```
Download our repository:
```
git clone https://github.com/kreimanlab/AugMem.git
```

## Preparing/indexing the datasets

This project uses three datasets. Each dataset has its own procedure for pre-processing and indexing (described below). In general, this involves the automated generation of a "dirmap" csv file for each dataset that indexes all of the images in the dataset - this dirmap is used to create train/test splits and select sequences/batches of images for training and testing under each of the 4 paradigms described in the paper (iid, class_iid, instance, class_instance) - this information will be generated in directories (one per dataset) named "<datasetname>_task_filelists. Although each dataset has its own indexing procedure to produce <datasetname>_dirmap.csv, the same functions are used to process this csv and produce <datasetname>_task_filelists. The task_filelists are then used for training and testing by the shell scripts for each agent (i.e. AugMem and a variety of baseline agents like EWC, iCARL, etc) found in the "scripts" folder. 

### Core50 dataset

1. Download the Core50 dataset from [this page](https://vlomonaco.github.io/core50/).
2. Unlike with the other two datasets below, the pre-generated core50_dirmap.csv provided with this repo can be used out-of-the-box. From the root project directory, run "sh scripts/setup_tasks_core50.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "core50_task_filelists"

### Toybox dataset

1. Download all three parts of the Toybox dataset from [this page](https://aivaslab.github.io/toybox/). Create a Toybox dataset directory called "toybox" (or whatever you like) and extract all three of the downloaded dataset components into this directory. Your Toybox dataset directory should contain the directories "animals", "vehicles", and "households"
2. Move the pre-generated toybox_dirmap.csv out of the "dataloaders" directory to a safe place. You still need to generate a new copy of this file yourself (see below), but a pre-generated copy is provided so that you can see what it's supposed to look like. 
3. Run the toybox_dirmap.py script to extract frames from the Toybox dataset videos and generate the dirmap csv to index them.  This script requires ffmpeg, which it uses to extract images from the videos in the Toybox dataset at a rate of 1 fps. This script will take several hours to run. Navigate to the "dataloaders" folder in the command line, and run "python toybox_dirmap.py <dataset_path>", replacing <dataset_path> with the location of your Toybox dataset directory. There should now be a "toybox_dirmap_unbalanced.csv" in the "dataloaders" folder, and in your Toybox dataset directory there should now be a new directory called "images"
4. Run the toybox_sample.py script to sample a slightly reduced version of the dataset with balanced statistics (i.e. guaranteed exactly the same number of objects per class, images per object, etc). This script may take several minutes to run. Navigate to the "dataloaders" directory, and run "python toybox_sample.py". There should now be a "toybox_dirmap.csv" in the "dataloaders" folder.
5. (Recommended) Verify that the dataset is correctly balanced by navigating to the "dataloaders" directory and running "python dirmap_csv_stats.py toybox_dirmap.csv". You should see that there are exactly 4350 examples (images) in each of the 12 classes, that each session has exactly 15 images (avg/min/max all equal 15), that each object has exactly 10 sessions, and that each class has exactly 29 objects. 
6. From the root project directory, run "sh scripts/setup_tasks_toybox.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "toybox_task_filelists"

### iLab-2M-Light dataset

1. Download the iLab-2M-Light dataset from [this page](https://bmobear.github.io/projects/viva/). Direct download link [HERE](http://ilab.usc.edu/ilab2m/iLab-2M-Light.tar.gz). Extract the dataset into a directory of your choice.
2. Move the pre-generated ilab2mlight_dirmap.csv out of the "dataloaders" directory to a safe place. You still need to generate a new copy of this file yourself (see below), but a pre-generated copy is provided so that you can see what it's supposed to look like. 
3. Run the ilab2mlight_dirmap.py script to generate the dirmap csv indexing the redistributed images in the dataset. This script should only take a few minutes to run. Navigate to the "dataloaders" folder in the command line, and run "python ilab2mlight_dirmap.py <dataset_path>", replacing <dataset_path> with the location of your iLab dataset directory. There should now be an "ilab2mlight_dirmap_all.csv" in the "dataloaders" folder
4. Run the ilab2mlight_sample.py script to sample a slightly reduced version of the dataset with balanced statistics (i.e. guaranteed exactly the same number of objects per class, images per object, etc). This script may take several minutes to run. Navigate to the "dataloaders" directory, and run "python ilab2mlight_sample.py". There should now be an "ilab2mlight_dirmap_massed.csv" in the "dataloaders" folder.
5. (Recommended) Verify that the dataset is correctly balanced by navigating to the "dataloaders" directory and running "python dirmap_csv_stats.py ilab2mlight_dirmap.csv". You should see that there are exactly 3360 examples (images) in each of the 14 classes, that each session has exactly 15 images (avg/min/max all equal 15), that each object has exactly 8 sessions, and that each class has exactly 28 objects. 
6. Distribute the images in the dataset to a nested directory structure by running the ilab2mlight_distribute_img_dirs.py script. The dataset comes by default with all of the images massed together in one directory, and this can make loading the data very slow during training. Navigate to the "dataloaders" folder and run "python ilab2mlight_distribute_img_dirs.py <dataset_path> <distributed_dataset_path>". The <distributed_dataset_path> should be a path to a new directory in which the distributed version of the dataset will be placed. Make sure you have enough room on your HDD/SSD before running this script, as it will make a copy of all of the sampled iamges in the dataset. This script will take several hours to run (e.g. maybe 12 hours). When it's finished, you should have "ilab2mlight_dirmap.py" in the "dataloaders" directory.
7. From the root project directory, run "sh scripts/setup_tasks_ilab2mlight.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "ilab2mlight_task_filelists"

## Running grid search for optimal hyperparameters for each algorithm
Skip this step if you want to run the optimal set of hyperparameters for each algorithm in our datasets. We use grid search on toybox as an example. Run the following to perform grid search:
```
cd gridsearch
#dataset, GPU id 0, GPU id 1
./toybox_setup_grid_raw.sh toybox 1 3
./toybox_gridsearch_raw.sh
cp -r  toybox_gridsearch_outputs/ ../
cp -r gridsearches ../scripts/
#resolve permission denied error when running generated shell scripts
chmod -R +x ../scripts
mv ../scripts/gridsearches ../scripts/gridsearches_toybox
#manually remove "--validate" from iCARL
./scripts/combined_gridsearch_toybox.sh
./scripts/combined_gridsearch_toybox_gpu2.sh
./toybox_setup_grid_raw.sh toybox 1 3
./toybox_gridsearch_raw.sh
cp -r  toybox_gridsearch_outputs/ ../
cp -r gridsearches ../scripts/
chmod -R +x ../scripts 
mv ../scripts/gridsearches ../scripts/gridsearches_toybox
#manually remove "--validate" from iCARL
./scripts/combined_gridsearch_toybox.sh
./scripts/combined_gridsearch_toybox_gpu2.sh
```
All grid search results will be in the form of ```test.csv``` and be stored in ```toybox_gridsearch_outputs``` folder. One can use ```summarize_gridsearch.py``` to plot the optimal set of hyperparameters for each algorithm.

## Running algorithms
Make sure that one is at the root of the repository. Run the following to train and save results of individual algorithm on CoRE50 dataset:
```
./scripts/optimal_core50/AugMem.sh core50 0
```
***NOTE*** Set ```DATAROOT``` in each shell script to the directory where the dataset is downloaded and stored.
Each shell script is an algorithm. It takes two input arguements: 
 - the dataset names: core50, toybox, ilab2mlight
 - the GPU ID to run the jobs: e.g. 0
 
If one wants to run all algorithms at once for a particular dataset:
```
./scripts/combined_optimal_core50.sh
```
Once the algorithms finish running, a ```*/test.csv``` file will be saved to store test results in ```core50_outputs``` folder.

To visualize the augmented memory and learnt hypothesis, enable ```--visualize``` flag in ```optimal_core50/AugMem.sh''' as below:
```
python -u experiment_aug.py --scenario class_iid --replay_coef 5 --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR  --visualize --first_times 1 --replay_times 1 --reg_coef 1000 --n_epoch 1  --memory_size 200 --freeze_feature_extract --n_runs 10 --model_type resnet --model_name ResNet18 --pretrained --memory_Nslots 100  --memory_Nfeat 8 --agent_type aug_mem --agent_name AugMem  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/AugMem_ResNet18/log.log
```

Run shell scripts in ```ablation``` folder for ablated models on CoRE50 dataset.

Similarly, one can run shell scripts stored in ```optimal_toybox``` and ```optimal_ilab``` folder on the other two datasets.

If one wants to only re-produce our results reported in the paper, all the results can be directly downloaded [HERE](https://drive.google.com/drive/folders/1Y0mM_jJiqBrZkChdKGTEb1cXhqBsjnUS?usp=sharing)
 
## Plotting results
All results can be plotted using Matlab scripts in ```matlab``` folder.
To plot the averaged accuracy as a function of task number, run ```PlotAccuracy_gridsearch_core50.m```.

To plot the accuracy in the first task as a function of task number, run ```PlotAccuracy_gridsearch_core50_1sttask.m```.

To visualize the t-sne clusterring, run ```VisRepresentsTSNE.m```

To visualize the learnt hypothesis projected to 2D image space, run ```VisAttentionBackImage.m``` followed by ```PlotMontage.m```. 

## Updates
 
1. We conducted two additional ablation studies: replace SqueezeNet with MobileNetV2 as backone, replace random sampling for replay buffer with herding
 (see ```Other_Ablated_AugMem``` folder and ```matlab/*_aaai.m``` files for result plots)
 
2. We added five additional baseline comparisions including: BIC, GSS, StableSGD, COPE, LwF
 (see ```other_models``` folder)
 
## Notes

The source code is for illustration purpose only. Path reconfigurations may be needed to run some MATLAB scripts. We do not provide techinical supports but we would be happy to discuss about SCIENCE!

## License

See [Kreiman lab](http://klab.tch.harvard.edu/code/license_agreement.pdf) for license agreements before downloading and using our source codes and datasets.


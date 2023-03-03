



source ~/.bashrc
conda activate Stereo


# Meta... save visualization here
YOUR_SAVE_PATH='/bask/projects/j/jiaoj-3d-vision/Hao/VideoHandler/SSL_Sound/save_path'


# Visualize the data with in here:
# /bask/projects/j/jiaoj-3d-vision/360XProject/Data/Meta/vis.csv

# visualizing the ITD prediction of videos over time
./scripts/visualization_video.sh 'YourVideo' $YOUR_SAVE_PATH




# Stereo Vision 
## Overview 
A. Feature extraction and matching
# ![1](https://github.com/advaitp/Depth-Estimation-using-Stereo-Camera/blob/main/images/feature.png)

B. Estimating Fundamental Matrix

C. Estimating Essential Matrix

D. Estimating Camera Poses
# ![2](https://github.com/advaitp/Depth-Estimation-using-Stereo-Camera/blob/main/images/epipolelines.png)

E. Rectification
# ![3](https://github.com/advaitp/Depth-Estimation-using-Stereo-Camera/blob/main/images/rectify.png)

F. Correspondence

Disparity Maps 
# ![4](https://github.com/advaitp/Depth-Estimation-using-Stereo-Camera/blob/main/images/disparity.png)

Depth Maps
# ![5](https://github.com/advaitp/Depth-Estimation-using-Stereo-Camera/blob/main/images/depth.png)

Requirements 3 folders curule, pendulum and octagon
DirPath: base path where data files exist
flag : flag to get the info about the folder
scaling : scaling percent where data files exist
windowsize : window size for disparity

flag is 3 for curule
flag is 1 for pendulum
flag is 2 for octagon


## Output
```
4 output files in the same directory
DisparityHeat{flag}.png Heat map of Disparity
DepthHeat{flag}.png Heat map of Depth
DisparityGray{flag}.png Gray image of Disparity
DepthGray{flag}.png Gray image of Depth
```

# To run the python code
```
python stereo.py
```


# rayleigh-eigendirections
Rayleigh EigenDirections (REDs) main repository.

*./example-images/stylegan2_hair.jpg*: Example of changing hairstyle while holding geometry and identity fixed. The seed image is the leftmost column, and each row is a different trajectory produced by our algorithm:
![Changing hairstyle](https://github.com/balakg/rayleigh-eigendirections/blob/main/example-images/stylegan2_hair.jpg?raw=true)


*./example-images/biggan-freq.jpg*: Example of fixing high spatial frequencies and changing low spatial frequencies. The seed image is the leftmost column, and each row is a different trajectory produced by our algorithm:
![Changing hairstyle](https://github.com/balakg/rayleigh-eigendirections/blob/main/example-images/biggan_freq.jpg?raw=true)


# Download external models
1. Download [stylegan2-ffhq-config-f.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl) from the NVIDIA stylegan2 repository and place file into stylegan2/. 

2. Download [BigGAN model](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view) and place folder (called '138k') into BigGAN-PyTorch/weights.

3. If you are using face segmentation, download this model:
https://drive.google.com/file/d/1o3pw4LzT8MtsE0oOzakvpaFc2InW58VD/view?usp=sharing and place file into face-parsing/.

4. If you are using face recognition, download this model:
https://drive.google.com/file/d/1x1WU62deZppzwUWitK3K2n1vmvMddTJM/view?usp=sharing and place file into insightface/.

5. If you are using 3D face landmarks, install [MediaPipe](https://pypi.org/project/mediapipe/).

# Other Requirements
1. See StyleGAN2 system [requirements](https://github.com/NVlabs/stylegan2). Tensorflow 1.14/1.15 and Cuda 10 toolkits are required. 

# Running Algorithm
To run the algorithm, run *main.py* with appropriate arguments. *reproduce_examples.sh* shows how to create the images in ./example-images/. Run:
```.bash
 bash ./run_main.sh 
```
Currently, *main.py* expects two GPUs as command line arguments, one for the GAN and one for evaluating features. Fixed and changing features are specified as strings -- example string specifications are given below.

You will find image results and corresponding latent codes saved to the results directory. If you also need feature distances for further analysis, you can save them by running *run_save_distances.sh* with the appropriate feature functions and experiments selected. More details on this will be added soon. 

## String specifications for features
Each feature string starts with one of four keys: id (identity), r (region), l (landmarks), a (attributes). We give some example strings below:

1. "id": identity
2. "r_head_seg": pixels within head region according to segmentation network
3. "r_eyes_coord": pixels within bounding box around eyes (fixed coordinates defined by LowRankGAN work)
4. "r_no-head_seg": pixels outside of the head region 
5. "r_all": all pixels
6. "r_head_seg_lo_2": pixels within head region + low pass filter using gaussian filter, sigma = 2
7. "r_head_seg_hi_2": pixels within head region + high pass filter using gaussian filter, sigma = 2
8. "l_all": all facial landmarks (3D coordinates)


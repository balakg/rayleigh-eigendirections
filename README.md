# rayleigh-eigendirections
Rayleigh EigenDirections (REDs) main repository.


# Download external models
1. Download [stylegan2-ffhq-config-f.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl) from the NVIDIA stylegan2 repository and place file into stylegan2/. 

2. Download [BigGAN model](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view) and place the folder (called '138k') into BigGAN-PyTorch/weights.

3. If you are using face segmentation, download this model:
https://drive.google.com/file/d/1o3pw4LzT8MtsE0oOzakvpaFc2InW58VD/view?usp=sharing and move the file into face-parsing/.

4. If you are using face recognition, download this model:
https://drive.google.com/file/d/1x1WU62deZppzwUWitK3K2n1vmvMddTJM/view?usp=sharing and move the file into insightface/.

5. If you are using 3D face landmarks, install [MediaPipe](https://pypi.org/project/mediapipe/).

# Other Requirements
1. See StyleGAN2 system [requirements](https://github.com/NVlabs/stylegan2). Tensorflow 1.14/1.15 and Cuda 10 toolkits are required. 

# Run REDs algorithm
Set variables in run_main.sh. Right now, the script expects two GPUs as command line arguments, one for the GAN
and one for evaluating features. E.g., run:
```.bash
 bash ./run_main.sh 0 1 
```

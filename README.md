# rayleigh-eigendirections
Rayleigh EigenDirections (REDs) main repository.


# Download external models
1. Download stylegan2-ffhq-config-f.pkl from the NVIDIA stylegan2 repository
and place file into stylegan2/. 

2. If you are using face segmentation, download this model:
https://drive.google.com/file/d/1o3pw4LzT8MtsE0oOzakvpaFc2InW58VD/view?usp=sharing and move the file into face-parsing/.

3. If you are using face recognition, download this model:
https://drive.google.com/file/d/1x1WU62deZppzwUWitK3K2n1vmvMddTJM/view?usp=sharing and move the file into insightface/.

# Run REDs algorithm
Set variables in run_main.sh. Right now, the script expects two GPUs as command line arguments, one for the GAN
and one for evaluating features. E.g., run:
```.bash
 bash ./run_main.sh 0 1 
```

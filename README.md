# Image Registration
## Author - Brendon Tankwa

The following outlines a description of a solution to the Image Registration problem for 2 georeferenced images
with different spectral bands.


## 1. File list

ImageRegistrationPT1.py             Image registration implementation
ImageClass.py                              Image class used to create image objects in image registration implementation
Readme                                        This file



## 2. Design

a. Style
Code is written with specifications for each function. Snake case naming convention for variables

b. Image registration algorithm
Each image is loaded then the correct image is resized to match the dimensions of the distorted image.
Features are detected then extracted from each image. Features are matched and then matched features are filtered
based on meters between gps between matched features. Filtered matches used to register image using Gdal warp with
gcp inputs



## 3. Running image registration

ImageRegistrationPT1.py contains main function. Function rubber_duck is run from main function.
```
Input parameters
:param path_correct_image: path to correct image
:param path_distorted_image: path to misaligned image
:param path_output_image: path to misaligned image
:param feature_detector: string that represents the detector to be used - ORB , BRISK , KAZE
```


## 4. Image registration analysis and decisions made

a. Feature detector used
Various detectors were available for this problem. OpenCV's ORB , BRISK , KAZE are most easily available.
SIFT and SURF were also recommended but were inaccessible due to patent protection, hence unavailable on OpenCV.
Online research revealed that ORB and BRISK are both largely considered to be more efficient than SIFT and SURF.

b. Resizing image
Resizing the images so that they have the same dimensions yielded better feature detection and matching, hence better
registration.

c. Filtering matches
Matched features were filtered based on meters between gps between matched features.
Outlying features (features whose distance does not lie within the standard deviation of the mean of meters between gps
are removed)

d. Increasing pixel intensity for grayndvi image
Each pixel in array for grayndvi image lies between -1 and 1 so lambda function used (x+1)*128 to convert to 256 px



## 5. Directions for future research

This program provides a solution to the registration problem of 2 images taken at different spectral bands only. More work
still needs to be done for registering images taken under different conditions (time, weather, luminosity etc.)
This program was tested using cases with small distortions and results for images with large distortions are still
inconclusive. No metric to meature this degree of distortion yet.

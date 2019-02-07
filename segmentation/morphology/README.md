# Retina_Vessel_Segmentation
This code is retina vessel segmentation practice. I will implement retinal vessel segmentation using Python and C++ 

Database : DRIVE, own Data

CapillDetector and PostProcess are not finished.

Just use Preprocessing and VenulesDetector.
Preprocess is composed of BlobImage, BlackRingRemoval, GrayScaleTrans.

First, BlobImage function is noise filtering process.
  The function has median, Gaussian, Bilateral filters.
  So, you just select(should remove the comment!) the filter. I recommend the median filter
  
Second, BlackRingRemoval function remove the background.
  This function use the hough circle transform.
  find the circle line(bcz background and non-background have different intensity. you can find the edge!)
  and replace background to average of center of retina image.
  
Third, Gray scale transforming.
  
VenuledDetector
  Use Top-hat morphological filter.
  
  
If you have any question, you can send this email "qhsh9713@gmail.com"
  
  

## Retinal vessel feature extraction

### Features

* Skeleton
* Brunch point (intersection)
* End point
* Diameter
* Angle



----

### Feature extract method

#### 1. Centerline extraction

####  [Skeleton](https://en.wikipedia.org/wiki/Topological_skeleton)

The skeleton usually emphasizes geometrical and topological properties of the shape, such as its [connectivity](https://en.wikipedia.org/wiki/Connectedness), [topology](https://en.wikipedia.org/wiki/Topology), [length](https://en.wikipedia.org/wiki/Length), [direction](https://en.wikipedia.org/wiki/Direction_(geometry)), and [width](https://en.wikipedia.org/wiki/Width).  => **That's why we should extract vessel skeleton.**

![image-20190225165904912](/Users/hyeonwoojeong/Desktop/bonoSpace/typoraImg/image-20190225165904912.png) 
![image-20190225165935456](/Users/hyeonwoojeong/Desktop/bonoSpace/typoraImg/image-20190225165935456.png)
**figure1. DRIVE train image and groundtruth**

we can get the segmented vessel image from vessel segmentation algorithm.
segmented vessel images are source of skeletonize.



#### [Pruning](https://en.wikipedia.org/wiki/Pruning_(morphology))

It is used as a complement to the [skeleton](https://en.wikipedia.org/wiki/Topological_skeleton) and thinning algorithms to remove unwanted parasitic components (spurs). 


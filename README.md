##Goals of Juggle Tracker
###Main Goals
* Track juggling balls
* Count catches
* Calculate siteswaps
* Keep track of which hand balls are in

###Other Goals
* Real time juggling analysis
* Background subtraction
* Ball color calibration
  * Works by making circles of regions to get the color of and regions to ignore.
  * Click once to set the center of the circle and then click again to set the radius.
    * Left clicking means you want to get the colors in the circle.
    * Right clicking means to ignore the area in the circle when considering background colors. Primarily used to circle partially covered balls when doing a smart color matching algorithm.
  * Types of color selecting algorithms:
    * Naive: Uses smallest rectangle possible to capture EVERY point circled. Prone to also picking up backgroud of similar color.
    * Smart: Tries to fit the rectangle to include as many circled points as possible and s little background as possible.
* Non-ball props

##Hurdles
* Coming up with a good, general algoritm for identifying catches and throws is difficult
  * Particularly challenging for 1 throws since their nature doesn't give them the arc of other throws
* Backgrounds with similar colors, which will have to be subtracted, but are currently eroded out

##Dependencies
* OpenCV for python (cv2)
* numpy (needed for OpenCV)
* scipy (for k-means clustering done in calibration)

##Demo and Current Status (as ofJuly 23, 2014)
[![Juggle Tracker Demo](http://img.youtube.com/vi/SJMk1RfxAT8/0.jpg)](http://youtu.be/SJMk1RfxAT8)

##Authors
Kai Smith (kai.smith@case.edu)

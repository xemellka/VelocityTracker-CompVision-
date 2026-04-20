# simpleVelocityTracker(CompVision)

A simple computer vision program for estimating an object's velocity from two images. OpenCV detects the object, then we define real-world coordinates and their corresponding pixel coordinates. By using cv.findHomography, we determine the object's actual ground coordinates in the photos, compare the positions between the two frames, and calculate the speed

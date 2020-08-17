# Detecting Stovetop Status with OpenCV and ARUCO Marker

This demo serves as the first step towards constructing a vision-based smart reminder for vulnerable populations like older adults and people with visino impairments. An envisioned use case is that when the user forgets to turn off the stove, the smart reminder will be able to alert the user immediately. We can definitley extend the application to detecting the on/off state of a TV or the on/off state of a light switch. 

**Step 1: Place an ARUCO marker on the stove, ideally next to the stove knob**

<p align="center">
  <img src="images/stove_pc.JPG" width="300" height="500">
</p>

**Step 2: Shoot videos of different stove top status. One for stove-off and the other for stove-on**

**Step 3: Select regions of interest you would like the system to track**

**Step 4: Let the system process the video data**
Feature extraction all that. 32x32 feature space. I find it commonly used. 

**Step 5: Test the system on a newly shot video**
Two test cases. The system works well as long as there the video quality is clear, little occulusion. 

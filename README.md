# Multi Object Tracking From Scratch

This project implements a multi object tracker from scratch, using YOLOX as the detector. It combines ideas from SORT (2017), OC-SORT (2022), and ByteTrack (2021). 

### Demo:

_Wait for a second for the gif to load._

<div align="center">
<p>
<img src="img/MOT_test.gif" width="400"/> 
</p>
<br>
<div>

</div>

</div>

-----------------------------------------------------------------------------------------------------------------


TO DO:

* Add an appearance model using ideas from [Simple Cues Lead to a Strong Multi-Object Tracker (2022)](https://arxiv.org/pdf/2206.04656.pdf) and [Bag of tricks for ReID paper](https://arxiv.org/pdf/1903.07071.pdf). 
  * ~~Add active and inactive track handling.~~
  * Rework code to make it easier to modify. Move the code from update into helper functions so update can be easier to modify.
  * ~~Use proxy distance averaged over all previous appearances for inactive tracks.~~
  * ~~Use long term memory handling.~~
  * Identify the thresholds for average distance between positive and negative classes. Use this to form the matching threshold for active and inactive tracks.
  * ~~Use on the fly domain adaptation for the appearance model where we use mean and variance for the current batch even during inference.~~
* Simple updates to make it better
  * Add NSA Kalman Filter by weighting error based on detection confidence
  * Make the process and noise matrices dependent on time like in Strong SORT
  * Replace IoU with GIoU and see if it improves things.
  * Replace the hungarian algorithm using the Jonker-Volgenant method to make it faster.
  * Add Camera Motion Compensation from either StrongSORT or the other paper.



----------------------------------------------------------------------------------------------------------------


Acknowledgements:

I learned a lot from reading the *SORT* (2017), *ByteTrack* (2021), *OC-SORT* (2022), *BoT-SORT* (2022) and *Simple Cues Lead to a Strong Object Tracker* (2022) papers (and many others). I thank these authors for graciously providing their code online for everyone. All code here is my own but it follows these papers and I referred to the codebase when I wanted some clarification on the paper. I'd like to also thank Jenny Seidenschwarz, the first author of the simple cues paper, for clarifying "on the fly domain adaptation" for me.
  

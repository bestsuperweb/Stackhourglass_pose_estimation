Pose estimation using StackHourglass, LSTMs to model pose as a sequence

The general idea is that we want to experiment with determining whether modelling the dependencies between joint location will give us a better estimate of complete human pose.

Dataset : MPII Single Human Pose Dataset

Directory structure
The structure for this repository is as follows

data/
  pretrained_models/
  annot/
  images/
notebooks/

checkpoints/

src/
  models/

writeup/

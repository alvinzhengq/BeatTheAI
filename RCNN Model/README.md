
# BeatTheAI RCNN Model
This is the primary folder used to train the Doodle AI. It contains the dataset, exported model, labels, etc. Pretty much everything used during the process of training the model.  

Model was trained for 3 hours on a NVIDIA GeForce GTX 1050 Ti, final loss was at 0.002, tracked using tensorboard.

# Table of Contents(Folders):

Export: Exported version of the trained model

Images: Dataset used to train the model, with XML files  

Data: Contains CVS of the labels, and tfrecords  

Training: Config file used for the Faster RCNN model architecture, pbtxt with class and corresponding id, and checkpoints of the model generated during training  

Faster_rcnn_inception_v2_coco_2018_01_28: Contains model architecture used for training

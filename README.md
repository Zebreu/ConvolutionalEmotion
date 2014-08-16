ConvolutionalEmotion
====================

A deep convolutional neural network system for live emotion detection

How to run:

Python 2.7 64 bit and DeCAF (https://github.com/UCB-ICSI-Vision-Group/decaf-release/) needs to be installed, as well as numpy, opencv, pygame (only if you want to play the simple game that changes based on your facial expression) and sklearn. Works on Linux (tested on Ubuntu, with Anaconda).

Download Cohn-Kanade+ dataset.

Execute emotionclassification.py to produce features files from the dataset and to save a classifier.

Execute zengame.py.

Paper about the system available on arXiv.org: link pending

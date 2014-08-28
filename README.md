ConvolutionalEmotion
====================

A deep convolutional neural network system for live emotion detection

How to run:

Python 2.7 64 bit and DeCAF (https://github.com/UCB-ICSI-Vision-Group/decaf-release/) needs to be installed, as well as numpy, opencv, pygame (only if you want to play the simple game that changes based on your facial expression) and sklearn. Works on Linux (tested on Ubuntu, with Anaconda).

Download Cohn-Kanade+ dataset. Not every sequence is annotated, so you can call remove() from emotionclassification.py to delete sequences without emotion labels.

Execute emotionclassification.py to produce features files from the dataset and to save a classifier. You might want to change the path to the Haar cascade files from OpenCV (the scripts assume they're in the same folder, so you can just copy them in there too).

Execute zengame.py.

Paper about the system available on arXiv.org: http://arxiv.org/pdf/1408.3750v1.pdf

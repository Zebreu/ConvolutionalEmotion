import os
import sys
import pickle

#import Image
import numpy
import shutil
from sklearn import svm
from sklearn import cross_validation
from decaf.scripts.imagenet import DecafNet
import cv2

mapping = "0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise"
mapping = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
labelnumbers = [45.0,18.0,59.0,25.0,69.0,28.0,83.0]

data_dir = "dataset location"

image_dir = "cohn-kanade-images"
label_dir = "Emotion"

number_sequences = 327
#feature_length = 4096
feature_length = 9216
"""
pool5_cudanet_out: the last convolutional layer output, of size 6x6x256.
fc6_cudanet_out: the 4096 dimensional feature after the first fully connected layer.
fc6_neuron_cudanet_out: similar to the above feature, but after ReLU so the negative part is cropped out.
fc7_cudanet_out: the 4096 dimensional feature after the second fully connected layer.
fc7_neuron_cudanet_out: after ReLU
"""

#feature_level = "fc6_neuron_cudanet_out"
#feature_level = "fc6_cudanet_out"
feature_level = "pool5_cudanet_out"

def getMoreFeatures():
    net = DecafNet()

    features = []
    labels = []
    counter = 0

    for participant in os.listdir(os.path.join(data_dir,image_dir)):
        for sequence in os.listdir(os.path.join(data_dir,image_dir, participant)):
            if sequence != ".DS_Store":
                image_files = sorted(os.listdir(os.path.join(data_dir,image_dir, participant,sequence)))
                cutoff = len(image_files)/2
                image_files = image_files[cutoff::]
                label_file = open(os.path.join(data_dir,label_dir, participant,sequence,image_files[-1][:-4]+"_emotion.txt"))
                label = eval(label_file.read())
                label_file.close()
                for image_file in image_files:
                    print counter, image_file
                    imarray = numpy.asarray(Image.open(os.path.join(data_dir,image_dir, participant,sequence,image_file)))
                    scores = net.classify(imarray, center_only=True)
                    features.append(net.feature(feature_level))
                    labels.append(label)
                    counter += 1

    numpy.save("featuresMore",numpy.array(features))
    numpy.save("labelsMore",numpy.array(labels))

def getPeakFaceFeatures():
    net = DecafNet()
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    features = numpy.zeros((number_sequences,feature_length))
    labels = numpy.zeros((number_sequences,1))
    counter = 0
    # Maybe sort them
    for participant in os.listdir(os.path.join(data_dir,image_dir)):
        for sequence in os.listdir(os.path.join(data_dir,image_dir, participant)):
            if sequence != ".DS_Store":
                image_files = sorted(os.listdir(os.path.join(data_dir,image_dir, participant,sequence)))
                image_file = image_files[-1]
                print counter, image_file
                imarray = cv2.imread(os.path.join(data_dir,image_dir, participant,sequence,image_file))
                imarray = cv2.cvtColor(imarray,cv2.COLOR_BGR2GRAY)
                rects = cascade.detectMultiScale(imarray, 1.3, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (150,150))
                if len(rects) > 0:
                    facerect=rects[0]
                    imarray = imarray[facerect[1]:facerect[1]+facerect[3], facerect[0]:facerect[0]+facerect[2]]
                scores = net.classify(imarray, center_only=True)
                features[counter] = net.feature(feature_level).flatten()
                label_file = open(os.path.join(data_dir,label_dir, participant,sequence,image_file[:-4]+"_emotion.txt"))
                labels[counter] = eval(label_file.read())
                label_file.close()
                counter += 1

    numpy.save("featuresPeakFace5",features)
    numpy.save("labelsPeakFace5",labels)


def getPeakFeatures():
    net = DecafNet()

    features = numpy.zeros((number_sequences,feature_length))
    labels = numpy.zeros((number_sequences,1))
    counter = 0
    # Maybe sort them
    for participant in os.listdir(os.path.join(data_dir,image_dir)):
        for sequence in os.listdir(os.path.join(data_dir,image_dir, participant)):
            if sequence != ".DS_Store":
                image_files = sorted(os.listdir(os.path.join(data_dir,image_dir, participant,sequence)))
                image_file = image_files[-1]
                print counter, image_file
                imarray = cv2.imread(os.path.join(data_dir,image_dir, participant,sequence,image_file))
                imarray = cv2.cvtColor(imarray,cv2.COLOR_BGR2GRAY)
                scores = net.classify(imarray, center_only=True)
                features[counter] = net.feature(feature_level)#.flatten()
                label_file = open(os.path.join(data_dir,label_dir, participant,sequence,image_file[:-4]+"_emotion.txt"))
                labels[counter] = eval(label_file.read())
                label_file.close()
                counter += 1

    numpy.save("featuresPeak5",features)
    numpy.save("labelsPeak5",labels)

def testClassifier():
    images = numpy.load("featuresPeakFace6.npy")
    labels = numpy.load("labelsPeakFace6.npy").flatten()
    
    #images = images.reshape((3018,4096)) # For featuresMore.npy

    if False: # If featureMores is used
        labels = labels.reshape((3018,1))
        imla = numpy.hstack((images,labels))
        numpy.random.shuffle(imla)
        images = imla[:,0:feature_length]
        labels = imla[:,-1]
    
    with open("oneversusone6.sav","w") as opened_file:
        for value in [1e1,1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
            #classifier = svm.LinearSVC(C=1e-4,class_weight="auto")
            #classifier = svm.SVC(C=value, kernel="poly", degree=6, class_weight="auto")
            classifier = svm.SVC(C=value, kernel="linear", class_weight="auto")

            if True:        
                # Leave-one-subject-out
                subjects = []
                subject_index = 0
                for participant in os.listdir(os.path.join(data_dir,image_dir)):
                    subjects.append(subject_index)
                    #print participant, subject_index
                    for sequence in os.listdir(os.path.join(data_dir,image_dir, participant)):
                        if sequence != ".DS_Store":
                            subject_index += 1

                print subjects, len(subjects)

                loso_results = []
                confusion_matrix = numpy.zeros((7,7))
                for i,subject in enumerate(subjects):
                    if i == len(subjects)-1:
                        trainimages = images[0:subject]
                        trainlabels = labels[0:subject]
                        testimages = images[subject:]
                        testlabels = labels[subject:]
                    else:
                        length = subjects[i+1]-subjects[i]
                        trainimages = numpy.vstack((images[0:subject],images[subject+length:]))
                        trainlabels = numpy.hstack((labels[0:subject],labels[subject+length:]))
                        testimages = images[subject:subject+length]
                        testlabels = labels[subject:subject+length]

                    classifier.fit(trainimages, trainlabels)
                    predictions = classifier.predict(testimages)
                    #print predictions,testlabels
                    for pre,lab in zip(predictions,testlabels):
                        confusion_matrix[int(lab)-1,int(pre)-1] += 1/labelnumbers[int(lab)-1]
                    loso_results.append(classifier.score(testimages,testlabels))
                    print i, loso_results[-1],

                final_score = sum(loso_results)/len(loso_results)
                print final_score
                total=0.0
                for i in xrange(7):
                    total+=confusion_matrix[i,i]

                total = total/7.0
                print total
                opened_file.write(str(total))
                
    with open("oneversusall6.sav","w") as opened_file:
        for value in [1e1,1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
            classifier = svm.LinearSVC(C=value,class_weight="auto")
            #classifier = svm.SVC(C=value, kernel="linear", class_weight="auto")

            if True:        
                # Leave-one-subject-out
                subjects = []
                subject_index = 0
                for participant in os.listdir(os.path.join(data_dir,image_dir)):
                    subjects.append(subject_index)
                    #print participant, subject_index
                    for sequence in os.listdir(os.path.join(data_dir,image_dir, participant)):
                        if sequence != ".DS_Store":
                            subject_index += 1

                print subjects, len(subjects)

                loso_results = []
                confusion_matrix = numpy.zeros((7,7))
                for i,subject in enumerate(subjects):
                    if i == len(subjects)-1:
                        trainimages = images[0:subject]
                        trainlabels = labels[0:subject]
                        testimages = images[subject:]
                        testlabels = labels[subject:]
                    else:
                        length = subjects[i+1]-subjects[i]
                        trainimages = numpy.vstack((images[0:subject],images[subject+length:]))
                        trainlabels = numpy.hstack((labels[0:subject],labels[subject+length:]))
                        testimages = images[subject:subject+length]
                        testlabels = labels[subject:subject+length]

                    classifier.fit(trainimages, trainlabels)
                    predictions = classifier.predict(testimages)
                    #print predictions,testlabels
                    for pre,lab in zip(predictions,testlabels):
                        confusion_matrix[int(lab)-1,int(pre)-1] += 1/labelnumbers[int(lab)-1]
                    loso_results.append(classifier.score(testimages,testlabels))
                    print i, loso_results[-1],

                final_score = sum(loso_results)/len(loso_results)
                print final_score
                total=0.0
                for i in xrange(7):
                    total+=confusion_matrix[i,i]

                total = total/7.0
                print total
                opened_file.write(str(total))

def getClassifier():
    images = numpy.load("featuresPeakFace5.npy")
    labels = numpy.load("labelsPeakFace5.npy").flatten()
    
    #images = images.reshape((3018,4096)) # For featuresMore.npy

    if False: # If featureMores is used
        labels = labels.reshape((3018,1))
        imla = numpy.hstack((images,labels))
        numpy.random.shuffle(imla)
        images = imla[:,0:feature_length]
        labels = imla[:,-1]
    
    # loso - 85.4 - 85.4 - 1e-4: 85.6\% - 85.5 - 83.4
    #classifier = svm.LinearSVC(C=1e-4,class_weight="auto")
    #classifier = svm.SVC(C=1e-6, kernel="poly", degree=2,class_weight="auto")
    classifier = svm.SVC(C=1e-6, kernel="linear", class_weight="auto")

    if True:        
        # Leave-one-subject-out
        subjects = []
        subject_index = 0
        for participant in os.listdir(os.path.join(data_dir,image_dir)):
            subjects.append(subject_index)
            print participant, subject_index
            for sequence in os.listdir(os.path.join(data_dir,image_dir, participant)):
                if sequence != ".DS_Store":
                    subject_index += 1

        print subjects, len(subjects)

        confusion_matrix = numpy.zeros((7,7))
        loso_results = []
        for i,subject in enumerate(subjects):
            if i == len(subjects)-1:
                trainimages = images[0:subject]
                trainlabels = labels[0:subject]
                testimages = images[subject:]
                testlabels = labels[subject:]
            else:
                length = subjects[i+1]-subjects[i]
                trainimages = numpy.vstack((images[0:subject],images[subject+length:]))
                trainlabels = numpy.hstack((labels[0:subject],labels[subject+length:]))
                testimages = images[subject:subject+length]
                testlabels = labels[subject:subject+length]

            classifier.fit(trainimages, trainlabels)
            predictions = classifier.predict(testimages)
            #print predictions,testlabels
            for pre,lab in zip(predictions,testlabels):
                confusion_matrix[int(lab)-1,int(pre)-1] += 1/labelnumbers[int(lab)-1]
            print numpy.round(confusion_matrix,3)*100
            loso_results.append(classifier.score(testimages,testlabels))
            print i, loso_results[-1]

        #with open("performance.sav","wb") as pickle_file:
        #    pickle.dump(loso_results, pickle_file)
        #print numpy.round(confusion_matrix,3)*100
        total=0.0
        for i in xrange(7):
            total+=confusion_matrix[i,i]

        print "Prediction:",total/7.0
        print sum(loso_results)/len(loso_results)
        #numpy.save("confusion_matrix.sav",confusion_matrix)

        #sets = cross_validation.LeaveOneOut(327)
        #results = cross_validation.cross_val_score(classifier, images, labels, cv=10)
        #print results, results.mean()
        
    else:
        classifier.fit(images,labels)
        with open("classifierBestFace.sav","wb") as pickle_file:
            pickle.dump(classifier, pickle_file)


def remove():
    a = 0
    for participant in os.listdir(os.path.join(data_dir,label_dir)):
        labelled = os.listdir(os.path.join(data_dir,label_dir,participant))
        images = os.listdir(os.path.join(data_dir,image_dir,participant))
        good = []
        for sequence in labelled:
            if len(os.listdir(os.path.join(data_dir,label_dir,participant,sequence))) != 0:
                good.append(sequence)
        for sequence in images:
            if not sequence in good:
                if sequence != ".DS_Store":
                    print "deleting image", participant, sequence
                    shutil.rmtree(os.path.join(data_dir,image_dir,participant,sequence))
        for sequence in labelled:
            if not sequence in good:
                if sequence != ".DS_Store":
                    print "deleting label", participant, sequence
                    shutil.rmtree(os.path.join(data_dir,label_dir,participant,sequence))

if __name__ == "__main__":
    #getPeakFeatures()
    #getMoreFeatures()
    getPeakFaceFeatures()
    getClassifier()
    #testClassifier()
    

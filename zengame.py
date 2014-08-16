'''
Created on 2012-10-03
@author: Sebastien Ouellet sebouel@gmail.com
'''
import sys
import os
import random
import math
import time
import threading
import Queue
import pickle
import scipy

import cv2
from decaf.scripts.imagenet import DecafNet

import pygame

# Global variables #

framerate = 60

white = (255,255,255)
black = (0,0,0)

width = 512
height = 512

mapping = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
#inverse_mapping = {"neutral":0, "anger":1, "contempt":2, "disgust":3, "fear":4, "happy":5, "sadness":6, "surprise":7}
rectangle_margin = 50
####################

class ComputeFeatures(threading.Thread):
    def __init__(self, net, array, nextarray, results):
        threading.Thread.__init__(self)
        self.net = net
        self.array = array
        self.results = results
        self.nextarray = nextarray
        with open("classifierBestFace.sav","rb") as opened_file:
            self.classifier = pickle.load(opened_file)
    
    def run(self):
        if True: # Live gaming loop
            while True:
                #time.sleep(1)
                #scores = self.net.classify(self.array)
                #self.result = net.top_k_prediction(scores,5)[1][0:5]
                if not self.nextarray.empty():
                    self.array = self.nextarray.get(True)
                    while not self.nextarray.empty():
                        self.nextarray.get()
                    #self.result = len(self.array)
                    scores = self.net.classify(self.array, center_only=True)
                    #prediction = self.classifier.predict(self.net.feature("fc6_neuron_cudanet_out"))
                    prediction = self.classifier.predict(self.net.feature("pool5_cudanet_out").flatten())
                    self.results.put(mapping[int(prediction)])
                    #self.results.put(self.net.top_k_prediction(scores,5)[1][0:5])
                    #print self.results.qsize()
        
        else: # Personalized training
            featureArray = numpy.zeros((300,4096))
            feature_index = 0
            while True:
                if feature_index > 299:
                    numpy.save("featureEmotion", featureArray)
                #time.sleep(1)
                #scores = self.net.classify(self.array)
                #self.result = net.top_k_prediction(scores,5)[1][0:5]
                if not self.nextarray.empty():
                    self.array = self.nextarray.get(True)
                    while not self.nextarray.empty():
                        self.nextarray.get()
                    #self.result = len(self.array)
                    scores = self.net.classify(self.array, center_only=True)
                    featureArray[feature_index]=self.net.feature("fc6_neuron_cudanet_out") #original
                    feature_index += 1
                    #self.results.put(self.net.top_k_prediction(scores,5)[1][0:5])
                    #print self.results.qsize()


class Unit(pygame.sprite.Sprite):
    """ Generic class for unit sprites """
    def __init__(self,pos,image=pygame.Surface([24,24])):
        pygame.sprite.Sprite.__init__(self)
        if not isinstance(image, pygame.Surface):
            self.image = load_image(image)
        else:
            self.image = image
        self.rect = self.image.get_rect()
        self.position = pos
        self.rect.center = self.position

    def update(self):
        self.rect = self.image.get_rect()
        self.rect.center = self.position

def main():
    net = DecafNet()

    video = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    arrays = Queue.LifoQueue()
    results = Queue.LifoQueue()

    detector = ComputeFeatures(net, [], arrays, results)
    detector.daemon = True
    detector.start()

    pygame.init()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('This is a video game')
    
    background = pygame.Surface((width,height))
    background.fill(white)
    screen.blit(background,(0,0))
    pygame.display.flip()
    
    allsprites = pygame.sprite.RenderUpdates()

    # Some parameters #
    
    size = 10
    enemy_surface = pygame.Surface((size,size))
    speed = 200.0
    playersize = 44

    # # # # # # # # # #

    player = Unit([256.0, 256.0], pygame.Surface((playersize,playersize)))
    allsprites.add(player)
    
    enemy_counter = 1.0
    clock = pygame.time.Clock()
    elapsed = 0.0
    accumulator = 0.0

    run = True
    face = None

    emotion_window = ["neutral","neutral","neutral","neutral","neutral","neutral"]
    #emotion_accumulator = 0.0
    current_emotion = "neutral"
    emotion = "neutral"
    health = 50
    game_time = 0.0

    while run:
        seconds = elapsed/1000.0
        accumulator += seconds
        game_time += seconds
        #emotion_accumulator += seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False

            #elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
            #    current = Unit((random.randint(0,512), random.randint(0,512)))
            #    allsprites.add(current)
            #elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
            #    for sprite in allsprites:
            #        sprite.position = [sprite.position[0]+random.randint(-5,5),sprite.position[1]+random.randint(-5,5)]

        if accumulator > enemy_counter: 
            allsprites.add(Unit([random.randint(0,512), 0],enemy_surface))
            accumulator = 0.0

        for sprite in allsprites.sprites():
            if sprite.image == enemy_surface:
                sprite.position[1]+=speed*seconds
            if sprite.position[1]>height-10:
                allsprites.remove(sprite)

        pressed = pygame.key.get_pressed()

        if pressed[pygame.K_RIGHT]:
            player.position[0]+=speed*seconds
        if pressed[pygame.K_LEFT]:
            player.position[0]-=speed*seconds
        if pressed[pygame.K_DOWN]:
            player.position[1]+=speed*seconds
        if pressed[pygame.K_UP]:
            player.position[1]-=speed*seconds           

        allsprites.update()

        allsprites.remove(player)
        health -= len(pygame.sprite.spritecollide(player,allsprites, True))
        allsprites.add(player)
        
        allsprites.clear(screen,background)
        changed = allsprites.draw(screen)
        pygame.display.update(changed)
        
        frame = video.read()[1]
        rects = cascade.detectMultiScale(frame, 1.3, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (150,150))

        #arrays.put(frame)

        # Idea: increase the size of the rectangle
        if len(rects) > 0:
            facerect = rects[0]
            #facerect[0] -= (rectangle_margin-30)
            #facerect[2] += rectangle_margin
            #facerect[1] -= (rectangle_margin-20)
            #facerect[3] += rectangle_margin
            face = frame[facerect[1]:facerect[1]+facerect[3], facerect[0]:facerect[0]+facerect[2]]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            arrays.put(face)
            if True:
                for (x,y,w,h) in rects:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        if not results.empty():
            emotion = results.get()
            emotion_window.append(emotion)
            emotion_window.pop(0)
            current_emotion = max(set(emotion_window),key=emotion_window.count)
            print "Current emotion:", current_emotion, "- Last detected:", emotion#, emotion_window
            if current_emotion == "happy":
                enemy_counter += 0.03
                enemy_counter = min(0.7,enemy_counter)
            else:
                enemy_counter += -0.02
                enemy_counter = max(0.01,enemy_counter)
            print "Health:", health, "- Time:", game_time

        if health < 1:
            run = False
            print "Game over! Score:", game_time

        if face != None:
            cv2.imshow("face",face)
        cv2.imshow("frame",frame)
        c = cv2.waitKey(1)
        if c == 27:
            cv2.destroyWindow("frame")
            cv2.destroyWindow("face")
            break

        elapsed = clock.tick(framerate)

    video.release()
    cv2.destroyAllWindows()

    pygame.quit()

if __name__ == "__main__":
    main()

        


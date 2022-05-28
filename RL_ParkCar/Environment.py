# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 01:12:49 2022

@author: nedim
"""
import Car
import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from pygame.math import Vector2
import os

class Environment:
    def __init__(self):
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = Car.Car()
        self.parkArea = Vector2(20.0,40.0)
        
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
       
        
    def step(self,dt,action):
        state_list = []
        
        # update
        self.agent.update(dt,action)
        self.reward = -1
        
        if self.agent.position == self.parkArea:
            self.reward = 150
            self.total_reward += self.reward
            self.done = True
            print("Total Reward : {}".format(self.total_reward))
        
        elif self.agent.position == Vector2(100,100):
            self.reward = -150
            self.total_reward += self.reward
            self.done = True
            print("Total Reward : {}".format(self.total_reward))

        # find distance
        state_list.append((self.agent.position-self.parkArea,self.agent.angle % 360,self.agent.steering))
        state_list.append(self.reward)
        state_list.append(self.done)

        return [state_list]
        #NORMALDE NEXTSTATE,REWARD,DONE,inf
        
        
    # reset
    def initialStates(self):
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent.position = Vector2(20, 20)
        self.agent.velocity = Vector2(0.0, 0.0)
        self.agent.angle = 0
        self.agent.steering = 0
        
        state_list = []

        state_list.append((self.agent.position-self.parkArea,self.agent.angle % 360,self.agent.steering))
        state_list.append(self.reward)
        state_list.append(self.done)
        
        return [state_list]
        
    
    def run(self):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        ppu = 32
        
        batch_size = 24
        state = self.initialStates()
        
        while self.done == False:
            
            dt = self.clock.get_time() / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
            
            # update
            action = self.agent.act(state)
            next_state = self.step(dt,action)
            self.total_reward += self.reward

            
            # storage
            #self.agent.remember(state, action, self.reward, next_state, self.done)
            
            # update
            #state = next_state
            
            # training
            #self.agent.replay(batch_size)
            
            #epsilon greedy
            #self.agent.adaptiveGreedy()
            
            # draw
            self.screen.fill((0, 0, 0))
            rotated = pygame.transform.rotate(car_image, self.agent.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, self.agent.position * ppu - (rect.width / 2, rect.height / 2))
            pygame.display.flip()

            self.clock.tick(self.ticks)

        pygame.quit()
        
if __name__ == "__main__":
    env=Environment()
    liste = []
    t = 0
    while True:
        #t += 1
        #print("Episode : ",t)
        #liste.append(env.total_reward)
        env.run()
        





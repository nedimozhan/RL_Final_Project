# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 01:17:57 2022

@author: nedim
"""

import os
import pygame
from math import sin, radians, degrees, copysign, cos
from pygame.math import Vector2
import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2


# This class will be an agent :)
class Car:
    def __init__(self):
        # Initialize parameters
        self.position = Vector2(20, 20)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = 0
        self.length = 2
        self.max_steering = 30
        self.steering = 0
        
        
        # STATES
        # ARABANIN PARK YERİNE GÖRE AÇISI
        # ARABANIN POZİSYONUNUN PARK YERİNE UZAKLIĞI
        # DİREKSİYON AÇISI
        self.state_size = 3
        
        # ACTIONS
        # DİREKSİYONU SAĞA ÇEVİR İLERİ GİT   
        # DİREKSİYONU SAĞA ÇEVİR GERİ GİT
        # DİREKSİYONU SOLA ÇEVİR İLERİ GİT
        # DİREKSİYONU SOLA ÇEVİR GERİ GİT
        # DİREKSİYONA DOKUNMA İLERİ GİT
        # DİREKSİYONA DOKUNMA GERİ GİT
        self.action_size = 6 # DİREKSİYONU SAĞA VEYA SOLA ÇEVİR,GERİ VEYA İLERİ
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()
    
    def build_model(self): #DEĞİŞECEKKKKK
        model = Sequential()
        model.add(Dense(48,input_dim=self.state_size,activation="relu"))
        model.add(Dense(self.action_size,activation="linear"))
        model.compile(loss="categorical_crossentropy",optimizer='adam')
        return model
    
    def remember(self,state,action,reward,next_state,done):
        #storage
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        state = np.array(state)
        #if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)
        #act_values = self.model.predict(state)
        #return np.argmax(act_values[0])
        
    def replay(self,batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        for state ,action,reward,next_state,done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target,verbose = 0)
        
    def adaptiveGreedy(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update(self, dt, action):
        print("Position : {}".format(self.position))
        
        if action == 1:       # DİREKSİYONU SAĞA ÇEVİR İLERİ GİT   
            self.steering -= 30 * dt
            self.velocity.y = -5;
        
        elif action == 2:     # DİREKSİYONU SAĞA ÇEVİR GERİ GİT
            self.steering -= 30 * dt
            self.velocity.y = 5;
        
        elif action == 3:     # DİREKSİYONU SOLA ÇEVİR İLERİ GİT
            self.steering += 30 * dt
            self.velocity.y = -5;
        
        elif action == 4:     # DİREKSİYONU SOLA ÇEVİR GERİ GİT
            self.steering += 30 * dt
            self.velocity.y = 5;
        
        elif action == 5:     # DİREKSİYONA DOKUNMA İLERİ GİT
            self.steering = 0
            self.velocity.y = -5;
        
        elif action == 6:     # DİREKSİYONA DOKUNMA GERİ GİT 
            self.steering = 0
            self.velocity.y = 5;
        
        self.steering = max(-self.max_steering, min(self.steering, self.max_steering))
        
        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = -self.velocity.y / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
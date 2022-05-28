import numpy as np
from gym.envs.classic_control import rendering
import os
from os import path
from PIL import Image


class CarModel:
    def __init__(self):
        self.info = 'Simple 2D car dynamic model'
        self.length = 2

        # posx,posy,velx,vely,angle,steering
        self.state = np.zeros(5, dtype=np.float64)
        self.observation = np.zeros(9)
        self.obs_dim = 9
        self.act_dim = 9
        self.state_dot = np.zeros(5)
        self.dt = 1

        self.x_lim = 21
        self.y_lim = 21
        self.viewer = None

        self.max_vel = 5
        self.max_steering = 30/180
        self.max_angle = np.pi

        self.reward = 0
        self.done = False
        self.step_count = 0
        self.max_step = 100

        self.target_state = np.zeros(4)

        self.vel_increase = 1
        self.steering_increase = 1

    def reset(self):
        initial_x = np.random.rand()*18+1
        initial_y = 0.
        initial_vely = 0.
        initial_angle = 0.
        initial_steering = 0.
        self.state = np.array(
            [initial_x, initial_y, initial_vely, initial_angle, initial_steering])

        target_y = 20.
        target_x = np.random.randint(1, 20)
        target_vely = 0.
        target_angle = 0.
        self.target_state = np.array(
            [target_x, target_y, target_vely, target_angle])

        self.reward = 0
        self.done = False
        self.step_count = 0
        self.max_step = 100

        self.__update_obs()
        return self.observation

    def step(self, u):
        self.step_count += 1
        u = self.__convert_u(u=u)
        self.__update_state(u=u)
        self.__check_state_limit()
        self.__update_obs()
        self.__check_done_reward()
        return self.observation, self.reward, self.done

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 600)
            self.viewer.set_bounds(-1, self.x_lim+1, -1, self.y_lim+1)
            # fname = path.join(path.dirname(__file__), "sprites/drone.png")
            mapsheet = Image.open(os.path.join(
                path.dirname(__file__), 'figs/mapsheet.png'))

            drone_path = os.path.join(path.dirname(
                __file__), 'figs/car.png')
            reward_path = os.path.join(
                path.dirname(__file__), 'figs/park_area.png')

            sprite_dict = dict()
            sprite_dict['D'] = mapsheet.crop((4*16, 0, 5*16, 1*16))
            sprite_dict['X'] = mapsheet.crop((7*16, 1*16, 8*16, 2*16))
            sprite_dict['O'] = mapsheet.crop((2*16, 0, 3*16, 1*16))
            sprite_dict['-'] = mapsheet.crop((2*16, 5*16, 3*16, 6*16))

            self.drone_transforms = []
            self.drones = []

            self.reward_transform = []
            self.render_rew = []

            for i in range(1):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(drone_path, 2, 3))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(1):
                self.reward_transform.append(rendering.Transform())
                self.render_rew.append(rendering.Image(reward_path, 2, 2))
                self.render_rew[i].add_attr(self.reward_transform[i])

        for i in range(1):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(
                self.state[0], self.state[1])

        for i in range(1):
            self.viewer.add_onetime(self.render_rew[i])
            self.reward_transform[i].set_translation(
                self.target_state[0], self.target_state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __update_state(self, u):
        u1 = u[0]
        u2 = u[1]

        v = self.state[2]
        theta = self.state[3]
        S = self.state[4]

        self.state_dot = np.array(
            [v*np.sin(theta), v*np.cos(theta), u1, v*np.sin(S)/self.length, u2])

        self.state += self.state_dot*self.dt

    def __update_obs(self):
        self.observation = np.array(
            self.state.tolist()+self.target_state.tolist())

    def __convert_u(self, u):
        if u == 0:
            return [self.vel_increase, self.steering_increase]
        if u == 1:
            return [self.vel_increase, 0]
        if u == 2:
            return [self.vel_increase, -self.steering_increase]
        if u == 3:
            return [0, self.steering_increase]
        if u == 4:
            return [0, 0]
        if u == 5:
            return [0, -self.steering_increase]
        if u == 6:
            return [-self.vel_increase, self.steering_increase]
        if u == 7:
            return [-self.vel_increase, 0]
        if u == 8:
            return [-self.vel_increase, -self.steering_increase]

    def __check_state_limit(self):
        self.state[2] = max(min(self.state[2], self.max_vel), -self.max_vel)
        self.state[4] = max(
            min(self.state[4], self.max_steering), -self.max_steering)
        self.state[3] = max(
            min(self.state[3], self.max_angle), -self.max_angle)

        self.state[0] = max(min(self.state[0], 21), -1)
        self.state[1] = max(min(self.state[1], 21), -1)

    def __check_done_reward(self):
        pose_err = np.sqrt(
            (self.state[0]-self.target_state[0])**2+(self.state[1]-self.target_state[1])**2)
        vel_err = abs(self.state[2]-self.target_state[2])
        angle_err = abs(self.state[3]-self.target_state[3])
        self.reward = -pose_err
        if pose_err < 1 and vel_err < 1 and angle_err < 5/180:
            self.done = True
            self.reward += 1000
        elif self.step_count >= self.max_step:
            self.done = True
        else:
            self.done = False

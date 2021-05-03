# Name- Gaurav Ajariwal
# UTA ID- 1001396273

from copy import deepcopy
import csv
import sys

import numpy as np
import random


class Grid:
    def __init__(self, arr, reward):
        self.reward = reward
        self.grid = arr
        self.init_grid()
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

    def init_grid(self):
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == '.':
                    self.grid[x][y] = Block(self.reward,False)
                elif self.grid[x][y] == 'X':
                    self.grid[x][y] = Block(0, False, is_wall = True)
                else:
                    self.grid[x][y] = Block(float(self.grid[x][y]), True)

class Block:
    def __init__(self, value, is_terminal, is_wall = False):
        self.reward = value
        self.is_terminal = is_terminal
        self.is_wall = is_wall

class Environment:
    def __init__(self, path, reward, gamma, K, learning_boundary):
        self.path = path
        self.reward = float(reward)
        self.gamma = float(gamma)
        self.number_of_moves = int(K)
        self.learning_boundary = float(learning_boundary)
        self.init_environment()
        self.rows = self.grid.rows
        self.cols = self.grid.cols

        # directions
        self.up = (-1, 0)
        self.down = (1, 0)
        self.left = (0, -1)
        self.right = (0, 1)

    def init_environment(self):
        with open(self.path, 'r') as f:
            data = list(csv.reader(f))
            self.data = data
            self.grid = Grid(self.data, self.reward)

    def __getitem__(self, tup):
        x,y = tup
        return self.grid.grid[x][y]
    
    def random_start_state(self):                               #need a random start state
        states = []
        for x in range(self.rows):
            for y in range(self.cols):
                if not self[x,y].is_terminal and not self[x,y].is_wall:
                    states.append((x,y))
        rand_state = states[random.randint(0, len(states) - 1)]
        return rand_state


def q_learning_update(env, old_state, action, new_state, Q, N):
    if env[new_state].is_terminal:
        Q[new_state][None] = env[new_state].reward
    if old_state is not None:
        if N[old_state][action] != 0:
            N[old_state][action] += 1
        else:
            N[old_state][action] = 1


        c = 1 / N[old_state][action]
        prev_q = Q[old_state][action]
        reward = env[old_state].reward
        max_utility = max(Q[new_state].values())
        value = ((1 - c) * prev_q) + c * (reward + env.gamma * max_utility)
        Q[old_state][action] = value

def f(state,action, Q, N, boundary):
    return 1 if N[state][action] < boundary else Q[state][action]

def get_action(env, state, Q, N):
    movements = [ env.up, env.down, env.left, env.right ]
    poss = []
    for x in movements:
        poss.append( f(state, x, Q, N, env.learning_boundary) )
    return movements[ int(np.argmax(poss)) ]

def calculate_new_state(env, old_state, action):
    up = {
        env.up: env.up,
        env.down: env.down,
        env.left: env.left,
        env.right: env.right
    }

    left = {
        env.up: env.left,
        env.down: env.right,
        env.left: env.down,
        env.right: env.up
    }

    right = {
        env.up: env.right,
        env.down: env.left,
        env.left: env.up,
        env.right: env.down

    }

    movements = [*[action for _ in range(8)], right[action], left[action]]
    action = movements[random.randint(0,9)]
    new_x = old_state[0] + action[0]
    new_y = old_state[1] + action[1]
    if new_x >= 0 and new_x < env.rows and new_y >= 0 and new_y < env.cols:
        if env[new_x, new_y].is_wall:
            new_x, new_y = old_state[0], old_state[1]
    else:
        new_x, new_y = old_state[0], old_state[1]
    return new_x, new_y

def agent_model_q_learning(env):
    movements = [env.up, env.down, env.left, env.right]
    Q = { (k,v) : { i: 0 for i in movements } for k in range(env.rows) for v in range(env.cols) }
    N = { (k,v) : { i: 0 for i in movements } for k in range(env.rows) for v in range(env.cols) }
    moves = 0

    while moves < env.number_of_moves:
        old_state = None
        action = None
        new_state = env.random_start_state()
        while True:
            q_learning_update(env, old_state, action, new_state, Q, N)
            if env[new_state].is_terminal:
                break
            action = get_action(env, new_state, Q, N)
            old_state = new_state
            new_state = calculate_new_state(env, old_state, action)
            moves += 1

    return Q

def q_learning(env_file, non_term_reward, gamma, num_of_moves, N_e):
        env = Environment(env_file, non_term_reward, gamma, num_of_moves, N_e)
        Q_learn = agent_model_q_learning(env)
        print('\nutilities: ')
        for i in range(env.rows):
            string = ''
            for j in range(env.cols):
                if env[i,j].is_terminal:
                    env[i,j].reward = round(env[i,j].reward, 3)
                    string += str(env[i,j].reward)
                    continue
                q = round(max(Q_learn[i,j].values()), 3)
                string += str(q) + ', '
            if string[-1] == ' ':
                string = string[:-2]
            print(string)

env_file = sys.argv[1]
non_term_reward = sys.argv[2]
gamma = sys.argv[3]
num_of_moves = sys.argv[4]
N_e = sys.argv[5]

q_learning(env_file, non_term_reward, gamma, num_of_moves, N_e)
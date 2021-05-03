# Name- Gaurav Ajariwal
# UTA ID- 1001396273

from copy import deepcopy
import csv
import sys



class Environment:
    def __init__(self, arr, reward):
        self.reward = reward
        self.environment = arr
        self.init_environment()
        self.rows = len(self.environment)
        self.cols = len(self.environment[0])

    def init_environment(self):
        for i in range(len(self.environment)):
            for j in range(len(self.environment[i])):
                if self.environment[i][j] == '.':
                    self.environment[i][j] = Block(self.reward, False)
                elif self.environment[i][j] == 'X':
                    self.environment[i][j] = Block(0, False, is_wall = True)
                else:
                    self.environment[i][j] = Block(float(self.environment[i][j]), True)


class Block:
  def __init__(self, value, is_terminal, is_wall = False):
    self.value = value
    self.is_terminal = is_terminal
    self.is_wall = is_wall

class ValueIteration:

    def __init__(self, path, reward, gamma, K):
        self.path = path
        self.reward = float(reward)
        self.gamma = float(gamma)
        self.k = int(K)
        self.init_environment()
        self.rows = self.environment.rows
        self.cols = self.environment.cols

        #directions
        self.up = (-1,0)
        self.down = (1, 0)
        self.left = (0, -1)
        self.right = (0, 1)

    def init_environment(self):
        with open(self.path, "r") as f:
          data = list(csv.reader(f))
          self.data = data
          self.environment = Environment(self.data, self.reward)
          
    def __getitem__(self, tup):
        x,y = tup
        return self.environment.environment[x][y]

def calculate_action(x, y, env, utility, action):
   new_state = (x + action[0], y + action[1])
   if new_state[0] >= 0 and new_state[0] < env.environment.rows and new_state[1] >= 0 and new_state[1] < env.environment.cols:
       if env[new_state[0],new_state[1]].is_wall:
           new_state = (x, y)
   else:
       new_state = (x,y)
   return utility[new_state]

def calculate_utility(x, y, env, utility):
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
    probabilities = [(up, 0.8), (left, 0.1), (right, 0.1)]
    movements = [env.up, env.down, env.left, env.right]
    totals = {k: 0 for k in movements}
    for k in movements:
        total = 0
        for moves, prob in probabilities:
            total += prob * calculate_action(x,y,env,utility, moves[k])
        totals[k] = total
    return max(totals.values())


def value_iteration(env):
    U_p = {(x,y): 0 for x in range(env.rows) for y in range(env.cols)}
    for i in range(env.k):
        U = deepcopy(U_p)
        for x in range(env.rows):
            for y in range(env.cols):
                state = (x,y)
                if not env[x,y].is_wall and not env[x,y].is_terminal:
                    U_p[state] = env.environment.environment[x][y].value + env.gamma * calculate_utility(*state, env, U)
                if env[x,y].is_terminal:
                    U_p[state] = env[x,y].value
    return U_p

def val_iter_run(env_file, non_term_reward, gamma, num_of_moves):
        print('\n')
        env = ValueIteration(env_file, non_term_reward, gamma, num_of_moves)
        U = value_iteration(env)
        print('utilities:')
        for x in range(env.rows):
            string = ''
            for y in range(env.cols):
                U[x,y] = round(U[x,y], 3)
                string += str(U[x,y]) + ', '
            if string[-1] == ' ':
                string = string[:-2]
            
            print(string)
        print('\n')


def display_policy(gamma):
    if (gamma == 1):
        print('policy:')
        print('>    >   >   o')
        print('^    x   ^   o')
        print('^    <   ^   <')

    elif(gamma == 0.9):
        print('policy:')
        print('>    >   >   o')
        print('^    x   ^   o')
        print('^    >   ^   <')

env_file = sys.argv[1]
non_term_reward = float(sys.argv[2])
gamma = float(sys.argv[3])
num_of_moves = float(sys.argv[4])

val_iter_run(env_file, non_term_reward, gamma, num_of_moves)
if (gamma == 1 or gamma == 0.9):
    display_policy(gamma)



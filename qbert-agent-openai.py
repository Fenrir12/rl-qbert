import gym
import numpy as np
import random as rd
import operator
from scipy import *


# Atrocious function to generate future locations depending on action
# WARNING !! May hurt your eyes
def gen_future_locs(locs):
    fut_locs = {}
    for n in range(21):
        fut_locs[str(n)] = 6 * [0]
    for idx, loc in enumerate(BOX_LOCS):
        # All future locs depending on position
        if idx == 0:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 'NOPE'
            fut_locs[str(idx)][3] = 2
            fut_locs[str(idx)][4] = 'NOPE'
            fut_locs[str(idx)][5] = 1
        elif idx == 1:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 0
            fut_locs[str(idx)][3] = 4
            fut_locs[str(idx)][4] = 'NOPE'
            fut_locs[str(idx)][5] = 3
        elif idx == 2:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 'NOPE'
            fut_locs[str(idx)][3] = 5
            fut_locs[str(idx)][4] = 0
            fut_locs[str(idx)][5] = 4
        elif idx == 3:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 1
            fut_locs[str(idx)][3] = 7
            fut_locs[str(idx)][4] = 'NOPE'
            fut_locs[str(idx)][5] = 6
        elif idx == 4:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 2
            fut_locs[str(idx)][3] = 8
            fut_locs[str(idx)][4] = 1
            fut_locs[str(idx)][5] = 7
        elif idx == 5:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 'NOPE'
            fut_locs[str(idx)][3] = 9
            fut_locs[str(idx)][4] = 2
            fut_locs[str(idx)][5] = 8
        elif idx == 6:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 3
            fut_locs[str(idx)][3] = 11
            fut_locs[str(idx)][4] = 'NOPE'
            fut_locs[str(idx)][5] = 10
        elif idx == 7:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 4
            fut_locs[str(idx)][3] = 12
            fut_locs[str(idx)][4] = 3
            fut_locs[str(idx)][5] = 11
        elif idx == 8:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 5
            fut_locs[str(idx)][3] = 13
            fut_locs[str(idx)][4] = 4
            fut_locs[str(idx)][5] = 12
        elif idx == 9:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 'NOPE'
            fut_locs[str(idx)][3] = 14
            fut_locs[str(idx)][4] = 5
            fut_locs[str(idx)][5] = 13
        elif idx == 10:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 6
            fut_locs[str(idx)][3] = 16
            fut_locs[str(idx)][4] = 21
            fut_locs[str(idx)][5] = 15
        elif idx == 11:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 7
            fut_locs[str(idx)][3] = 17
            fut_locs[str(idx)][4] = 6
            fut_locs[str(idx)][5] = 16
        elif idx == 12:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 8
            fut_locs[str(idx)][3] = 18
            fut_locs[str(idx)][4] = 7
            fut_locs[str(idx)][5] = 17
        elif idx == 13:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 9
            fut_locs[str(idx)][3] = 19
            fut_locs[str(idx)][4] = 8
            fut_locs[str(idx)][5] = 18
        elif idx == 14:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 22
            fut_locs[str(idx)][3] = 20
            fut_locs[str(idx)][4] = 9
            fut_locs[str(idx)][5] = 19
        elif idx == 15:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 10
            fut_locs[str(idx)][3] = 'NOPE'
            fut_locs[str(idx)][4] = 'NOPE'
            fut_locs[str(idx)][5] = 'NOPE'
        elif idx == 16:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 11
            fut_locs[str(idx)][3] = 'NOPE'
            fut_locs[str(idx)][4] = 10
            fut_locs[str(idx)][5] = 'NOPE'
        elif idx == 17:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 12
            fut_locs[str(idx)][3] = 'NOPE'
            fut_locs[str(idx)][4] = 11
            fut_locs[str(idx)][5] = 'NOPE'
        elif idx == 18:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 13
            fut_locs[str(idx)][3] = 'NOPE'
            fut_locs[str(idx)][4] = 12
            fut_locs[str(idx)][5] = 'NOPE'
        elif idx == 19:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 14
            fut_locs[str(idx)][3] = 'NOPE'
            fut_locs[str(idx)][4] = 13
            fut_locs[str(idx)][5] = 'NOPE'
        elif idx == 20:
            fut_locs[str(idx)][0] = idx
            fut_locs[str(idx)][1] = idx
            fut_locs[str(idx)][2] = 'NOPE'
            fut_locs[str(idx)][3] = 'NOPE'
            fut_locs[str(idx)][4] = 14
            fut_locs[str(idx)][5] = 'NOPE'
    return fut_locs


def is_yellow(px):
    if [210, 210, 64] in px:
        return True
    return False


def is_orange(px):
    if [181, 83, 40] in px:
        return True
    return False


def is_purple(px):
    if [146, 70, 192] in px:
        return True
    return False


def is_blue(px):
    if [45, 87, 176] in px:
        return True
    return False


# Gets position of snake
def snake_there(obs):
    pos = CHAR_LOCS
    # Look for bert location
    for idx, pix in enumerate(pos):
        if is_purple(obs[pix[1]][pix[0]]):
            return idx
    return S_LOC


# Gets position of Bert
def bert_there(obs):
    pos = CHAR_LOCS
    # Look for bert location
    for idx, pix in enumerate(pos):
        if is_orange(obs[pix[1] - 2][pix[0]]) or is_orange(obs[pix[1] - 2][pix[0] + 1]) or is_orange(
                obs[pix[1] - 4][pix[0]]):
            return idx
    return B_LOC


# Get feature colors of boxes
def color_features(obs, pos):
    features = 21 * [0.0]
    for idx, pix in enumerate(BOX_LOCS[0:20]):
        if is_yellow(obs[pix[1]][pix[0]]):
            features[idx] = 1
        if features[idx] != 1 and idx == pos:
            features[pos] = 1
    return sum(features) / 21


# Get features of distance with snake
def snakedis_feature(bert_fut_pos):
    if bert_fut_pos == 'NOPE':
        bert_pos = CHAR_LOCS[S_LOC]
    else:
        bert_pos = CHAR_LOCS[bert_fut_pos]
    snake_pos = CHAR_LOCS[S_LOC]
    dis = np.sqrt((bert_pos[1] - snake_pos[1]) ** 2 + (bert_pos[0] - snake_pos[0]) ** 2)
    return dis


# Get distance from blue box feature
def bluedistance_feature(obs, bert_fut_pos):
    if bert_fut_pos == 'NOPE':
        bert_pos = CHAR_LOCS[S_LOC]
    else:
        bert_pos = CHAR_LOCS[bert_fut_pos]
    distances = []
    for idx, pix in enumerate(BOX_LOCS[0:20]):
        if is_blue(obs[pix[1]][pix[0]]):
            distances.append(
                1 - np.sqrt((bert_pos[1] - BOX_LOCS[idx][1]) ** 2 + (bert_pos[0] - BOX_LOCS[idx][0]) ** 2) / 158)
    return max(distances)

# Lives feature
def get_lives(obs):
    lives = 0
    if is_yellow(obs[23][37]):
        lives += 1
    if is_yellow(obs[23][46]):
        lives += 1
    if is_yellow(obs[23][53]):
        lives += 1
    return lives


def predict(obs):
    features = []
    # Get future position
    fut_pos = FUTURE_POS[str(B_LOC)]
    for action, pos in enumerate(fut_pos):
        snakedis = snakedis_feature(pos) / 158
        colorsum = color_features(obs, pos)
        bluedis = bluedistance_feature(obs, pos)
        if pos == 'NOPE':
            lives = (LIVES - 1)/3
        else:
            lives = LIVES/3

        features.append([snakedis, colorsum, bluedis, lives])
    return features

def predict_next(obs, action):
    features = []
    # Get future position
    fut_pos = FUTURE_POS[FUTURE_POS[str(B_LOC)][action]]
    for action, pos in enumerate(fut_pos):
        snakedis = snakedis_feature(pos) / 158
        colorsum = color_features(obs, pos)
        bluedis = bluedistance_feature(obs, pos)
        if pos == 'NOPE':
            lives = (LIVES - 1)/3
        else:
            lives = LIVES/3

        features.append([snakedis, colorsum, bluedis, lives])
    return features


def gradient(features):
    return sum(features)


def get_Q(weights, features):
    Q = []
    for idx, feature in enumerate(features):
        Q.append(feature[0] * weights[0] + feature[1] * weights[1] + feature[2] * weights[2] + feature[3] * weights[3])
    return Q


def bert_on_box(obs):
    for loc in BOX_LOCS:
        if is_orange(obs[loc[1] - 2][loc[0]]) or is_orange(obs[loc[1] - 2][loc[0]+1]) or is_orange(obs[loc[1] - 4][loc[0]]):
            return True
    return False


BOX_LOCS = [[78, 38], [65, 66], [94, 66], [54, 95], [78, 95], [106, 95], [42, 124], [66, 124], [94, 124], [118, 124],
            [30, 153], [54, 153], [78, 153], [106, 153], [130, 153], [18, 182], [42, 182], [66, 182], [94, 182],
            [118, 182], [142, 182], [12, 139], [146, 139]]
CHAR_LOCS = [[77, 28], [65, 56], [93, 56], [53, 85], [77, 85], [105, 85], [41, 114], [65, 114], [93, 114], [117, 114],
             [29, 143], [53, 143], [77, 143], [105, 143], [129, 143], [17, 172], [41, 172], [65, 172], [93, 172],
             [117, 172], [141, 172], [12, 129], [146, 129]]
B_LOC = 0
S_LOC = 2
LIVES = 3
FUTURE_POS = gen_future_locs(BOX_LOCS)
# Initialize learning environment
env = gym.make('Qbert-v0')
env.reset()

# Learning hyperparameters
episodes = 1000  # how many episodes to wait before moving the weights
max_time = 10000
gamma = 0.99  # discount factor for reward
lr = 1e-4
weights = np.random.rand(4)
e = 0.15

for episode in range(episodes):
    observation = env.reset()
    total_reward = 0
    for time in range(max_time):
        env.render()
        # Select new actions and update with frame skipping
        if bert_on_box(observation):
            features = predict(observation)
            # Greedy policy
            if rd.random < e:
                action = env.action_space.sample()
            else:
                # policy max Q action
                Qs = get_Q(weights, features)
                Qs = [Qs[2], Qs[3], Qs[4], Qs[5]]
                Q = max(Qs)
                action = Qs.index(Q) + 2
            observation, reward, done, info = env.step(action)
            S_LOC = snake_there(observation)
            B_LOC = bert_there(observation)
            if FUTURE_POS[str(B_LOC)][action] == 'NOPE':
                LIVES -= 1

            # Update weights
            next_Qs = get_Q(weights, predict(observation))
            next_Qs = [next_Qs[2], next_Qs[3], next_Qs[4], next_Qs[5]]
            next_Q = max(next_Qs)
            for idx, _ in enumerate(weights):
                weights[idx] = weights[idx] + lr*(reward + gamma*(next_Q - Q)*gradient(features))
            total_reward += reward
        # Do noop action if not on box
        observation, reward, done, info = env.step(0)
        if done:
            print("Episode {}:".format(episode))
            print("  completed in {} steps".format(time + 1))
            print("  total_reward was {}".format(total_reward))
            print("New weights are " + str(weights))
            break
print('success')

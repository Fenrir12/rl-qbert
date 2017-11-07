from ale_python_interface import ALEInterface
import gym
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import sys
import time as t
from PIL import Image

WEIGHTS1 = [1.3826337386217185, 23.894746079161084, 8.801830487930047, 11.254706535442095, 0.5956519333495852, 8.779244143679769, 1.2142990476462545, 1.5014086491630236, 1.2340120376539887, 1.2536234329023972, 1.1109156466921406, -1.3385189077421555, 0.4091773262075074, 1.4591866846765025, 1.7628712271103488, 2.177067408798442, 0.38667275775135457, 1.249181200223059, 2.208181286057919, 1.2595264191424724, 1.690644813808155, 0.21153815086304964, 0.9419314708311681, 1.085455920333917, 1.372615691498354, 0.9592344002780562, 1.2591047488657916, 13.684806533175662, 13.138060227438961, 11.44460497846998, 16.383418276389474]

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

# return True if pixel is yellow
def is_yellow(px):
    if YEL_PIX in px:
        return True
    return False

# return True if pixel is orange
def is_orange(px):
    if [181, 83, 40] in px:
        return True
    return False

# return True if pixel is green
def is_green(px):
    if [36, 146, 85] in px:
        return True
    return False

# return True if pixel is purple
def is_purple(px):
    if [146, 70, 192] in px:
        return True
    return False

# return True if pixel is blue
def is_blue(px):
    if BLUE_PIX in px:
        return True
    return False


# Gets position of snake on the 0 to 21 boxes
def snake_there(obs):
    pos = CHAR_LOCS
    # Look for bert location
    for idx, pix in enumerate(pos):
        if is_purple(obs[pix[1]][pix[0]]):
            return idx
    return S_LOC


# Gets position of green blob on the 0 to 21 boxes
def blob_there(obs):
    pos = CHAR_LOCS
    # Look for bert location
    for idx, pix in enumerate(pos):
        if is_green(obs[pix[1]][pix[0]]):
            return idx
    return G_LOC


# Gets position of Bert on the 0 to 21 boxes
def bert_there(obs):
    pos = CHAR_LOCS
    # Look for bert location
    for idx, pix in enumerate(pos):
        if is_orange(obs[pix[1] - 2][pix[0]]) or is_orange(obs[pix[1] - 2][pix[0] + 1]) or is_orange(
                obs[pix[1] - 4][pix[0]]):
            return idx
    return B_LOC


# FEATURE : Get position feature of bert.
# In a vector of 23, the position of Bert is given value 1
def pos_features(obs, pos):
    features = 23 * [0.0]
    if pos == 'NOPE':
        return features
    features[pos] = 1
    return features


# FEATURE : Turns to 1 if Bert's position gets him in the void
def doom_feature(obs, pos):
    if pos == 'NOPE':
        return 0
    return 1


# FEATURE : Returns 1 if Bert is on an unconquered box
def color_features(obs, pos):
    features = 21 * [0.0]
    for idx, pix in enumerate(BOX_LOCS[0:21]):
        if is_yellow(obs[pix[1]][pix[0]]):
            features[idx] = 1
        if features[idx] != 1 and idx == pos:
            return 1
    return 0


# Returns the number of conquered boxes of the screen
def num_col_boxes(obs):
    features = 21 * [0.0]
    for idx, pix in enumerate(BOX_LOCS[0:21]):
        if is_yellow(obs[pix[1]][pix[0]]):
            features[idx] = 1
    return sum(features)


# FEATURE : Returns 1 if bert gets closer to the green character
def blob_feature(bert_fut_pos):
    if bert_fut_pos == 'NOPE':
        bert_pos = CHAR_LOCS[B_LOC]
    else:
        bert_pos = CHAR_LOCS[bert_fut_pos]
    blob_pos = CHAR_LOCS[G_LOC]
    dis = np.sqrt((bert_pos[1] - blob_pos[1]) ** 2 + (bert_pos[0] - blob_pos[0]) ** 2)
    pres_dis = np.sqrt((CHAR_LOCS[B_LOC][1] - blob_pos[1]) ** 2 + (CHAR_LOCS[B_LOC][0] - blob_pos[0]) ** 2)
    if dis < pres_dis:
        return 1
    return 0

# FEATURE : Returns 0 if the related disk is not present in the screen
def disk_features(obs):
    disk1 = 1
    disk2 = 1
    if [0, 0, 0] in obs[BOX_LOCS[21][1]][BOX_LOCS[21][0]]:
        disk1 = 0
    if [0, 0, 0] in obs[BOX_LOCS[22][1]][BOX_LOCS[22][0]]:
        disk2 = 0
    return [disk1, disk2]


# FEATURE : Returns 1 if bert gets closer to the snake
def snakedis_feature(bert_fut_pos):
    if bert_fut_pos == 'NOPE':
        bert_pos = CHAR_LOCS[B_LOC]
    else:
        bert_pos = CHAR_LOCS[bert_fut_pos]
    snake_pos = CHAR_LOCS[S_LOC]
    dis = np.sqrt((bert_pos[1] - snake_pos[1]) ** 2 + (bert_pos[0] - snake_pos[0]) ** 2)
    pres_dis = np.sqrt((CHAR_LOCS[B_LOC][1] - snake_pos[1]) ** 2 + (CHAR_LOCS[B_LOC][0] - snake_pos[0]) ** 2)
    if dis > pres_dis:
        return 0
    return 1


# FEATURE : Value 1 if Bert gets closer to an unconquered box
def bluedistance_feature(obs, bert_fut_pos):
    if bert_fut_pos == 'NOPE':
        bert_pos = CHAR_LOCS[B_LOC]
    else:
        bert_pos = CHAR_LOCS[bert_fut_pos]
    distances = []
    pres_distances = []
    for idx, pix in enumerate(BOX_LOCS[0:20]):
        if is_blue(obs[pix[1]][pix[0]]):
            distances.append(
                np.sqrt((bert_pos[1] - BOX_LOCS[idx][1]) ** 2 + (bert_pos[0] - BOX_LOCS[idx][0]) ** 2) / 158)
        if is_blue(obs[pix[1]][pix[0]]):
            pres_distances.append(
                np.sqrt((CHAR_LOCS[B_LOC][1] - BOX_LOCS[idx][1]) ** 2 + (
                CHAR_LOCS[B_LOC][0] - BOX_LOCS[idx][0]) ** 2) / 158)
    if len(distances) == 0:
        return 0
    mindis = min(distances)
    pres_dis = min(pres_distances)
    if mindis < pres_dis:
        return 1
    return 0


# Returns features of possible future states
def predict(obs):
    features = []
    # Get future position
    fut_pos = FUTURE_POS[str(B_LOC)]
    for action, pos in enumerate(fut_pos[2:6]):
        snakedis = snakedis_feature(pos)
        colorsum = color_features(obs, pos)
        bluedis = bluedistance_feature(obs, pos)
        lives = LIVES / 4
        disks = disk_features(obs)

        blobdis = blob_feature(pos)
        doom = doom_feature(screen, pos)
        pos_feat = pos_features(obs, pos)
        features.append([snakedis] + [colorsum] + [bluedis] + [lives] + disks + [blobdis] + [doom] + pos_feat)#[snakedis, colorsum, bluedis, lives])# + disks + [blobdis] + [doom] + pos_feat)
    return features

# Returns Q values of features with or without optimistic prior
def get_Q(weights, features):
    action = 0
    Qi = 0
    Q = []
    for feature in features:
        for id in range(NUM_FEATURES):
            Qi += feature[id] * weights[id]
        if [action, features[action]] in N:
            pos = N.index([action, features[action]])
            n = Na[pos]
        else:
            n = 1
        action += 1
        if n >= Ne:
            Q.append(Qi)
        else:
            if USE_OPTIMISTIC_PRIOR == True:
                Q.append(Qi + 1/n*100)
            else:
                Q.append(Qi)
        Qi = 0
    return Q



BOX_LOCS = [[78, 38], [65, 66], [94, 66], [54, 95], [78, 95], [106, 95], [42, 124], [66, 124], [94, 124], [118, 124],
            [30, 153], [54, 153], [78, 153], [106, 153], [130, 153], [18, 182], [42, 182], [66, 182], [94, 182],
            [118, 182], [142, 182], [12, 139], [146, 139]]
CHAR_LOCS = [[77, 28], [65, 56], [93, 56], [53, 85], [77, 85], [105, 85], [41, 114], [65, 114], [93, 114], [117, 114],
             [29, 143], [53, 143], [77, 143], [105, 143], [129, 143], [17, 172], [41, 172], [65, 172], [93, 172],
             [117, 172], [141, 172], [12, 129], [146, 129]]
# Initialize positions of character and game parameters
B_LOC = 0
S_LOC = 20
G_LOC = 20
CLEARED = 0
STAGE = 1

# Initialize vectors for optimistic priors calculation
Na = []
N = []
Ne = 5

#Gets user defined parameters
SEED = int(sys.argv[1])
USE_OPTIMISTIC_PRIOR = True if sys.argv[2] == 'use_prior' else False
SEE_SCREEN = True if sys.argv[3] == 'set_screen' else False

# Generate future positions of Bert dpeending on current position and action
FUTURE_POS = gen_future_locs(BOX_LOCS)

# Learning hyperparameters
episodes = 400  # how many episodes to wait before moving the weights
max_time = 10000
gamma = 0.99  # discount factor for reward
lr = 1e-4
NUM_FEATURES = 31
weights = [rd.random() for _ in range(NUM_FEATURES)]
e = 0.15 if USE_OPTIMISTIC_PRIOR == False else 0.00

# Initialize learning environment
ale = ALEInterface()
ale.setBool('sound', False)
ale.setBool('display_screen', SEE_SCREEN)
ale.setInt('frame_skip', 1)
ale.setInt('random_seed', SEED)
rd.seed(SEED)
ale.loadROM("qbert.bin")
ELPASED_FRAME = 0

# Possible positions of Bert in the RAM right beforetaking any action
MEM_POS = [[69, 77], [92, 77], [120, 65], [97, 65], [147, 53], [124, 53],
           [152, 41], [175, 41], [180, 29], [203, 29], [231, 16], [231, 41],
           [175, 65], [180, 53], [203, 53], [147, 77], [120, 93], [152, 65],
           [231, 65], [175, 93], [97, 93], [180, 77], [231, 93], [180, 105],
           [147, 105], [203, 77], [175, 77], [175, 117], [231, 117], [203, 129],
           [203, 105], [180, 129], [231, 141], [152, 117], [124, 77], [124, 105],
           [152, 93]]
learning = []
# Limits action set to UP RIGHT LEFT DOWN actions of ALE environment
actions = range(2, 6)
# Starts the learning episodes
for episode in range(episodes):
    total_reward = 0
    sup_reward = 0
    action = 0
    rewards = []
    ram = ale.getRAM()
    Q = 0
    last_action = 0
    last_Q = 0
    last_features = NUM_FEATURES * [rd.random()]
    BLUE_PIX = [45, 87, 176]
    YEL_PIX = [210, 210, 64]
    # Starts iterations of episode
    for time in range(max_time):
        # Get bert pos in RAM
        B_POS = [ram[33], ram[43]]
        # Get number of lives remaining
        LIVES = ale.lives()
        # last_ram = ram
        ram = ale.getRAM()
        screen = ale.getScreenRGB()
        # Updates position of characters
        S_LOC = snake_there(screen)
        B_LOC = bert_there(screen)
        G_LOC = blob_there(screen)
        # Bert ready to take new action at permitted position
        # and frame 0 of action taking
        if ram[0] == 0 and B_POS in MEM_POS and CLEARED == 0:
            features = predict(screen)
            # e-greedy exploration. Action is updated only at right frames.
            if rd.random() < e:
                action = rd.choice(actions) - 2
                Qs = get_Q(weights, features)
                Q = Qs[action]
            else:
                # policy max Q action
                Qs = get_Q(weights, features)
                Q = max(Qs)
                action = Qs.index(Q)
            # Update optimistic prior if used
            if [action, features[action]] in N:
                pos = N.index([action, features[action]])
                Na[pos] += 1
            else:
                N.append([action, features[action]])
                Na.append(1)
            # Take action
            ale.act(action + 2)
            # Gets last meaningful reward in stack of frames
            reward = max(rewards)
            if B_LOC == S_LOC or FUTURE_POS[str(B_LOC)] == None:
                sup_reward = -50
            else:
                sup_reward = 0
            for id in range(len(weights)):
                update = reward + sup_reward + gamma * Q - last_Q
                weights[id] = weights[id] + lr * update * last_features[id]

            # Update state, Q and action and resets rewards vector
            last_action = action
            last_features = features[last_action]
            last_Q = Q
            total_reward += reward
            rewards = []
        else:
            # Stack rewards of precedent frames to capture reward associated to right action
            rewards.append(ale.act(0))

        # Sets the stage as cleared if all boxes are conquered
        if num_col_boxes(screen) == 21 and CLEARED == 0:
            CLEARED = 1

        # Updates color check of is_yellow and is_blue functions of blocks for new stage
        if CLEARED == 1 and B_LOC == 0:
            STAGE += 1
            # Fill with color codes of boxes on each level
            if STAGE == 2:
                BLUE_PIX = [210, 210, 64]
                YEL_PIX = [45, 87, 176]
            elif STAGE == 3:
                BLUE_PIX = [182, 182, 170]
                YEL_PIX = [109, 109, 85]
            CLEARED = 0
        # Reset game and start new episode if bert is game over
        if ale.game_over():
            learning.append(total_reward)
            plt.xlabel('Episodes (n)')
            plt.ylabel('Total reward of episode')
            plt.plot(range(0, len(learning)), learning)
            plt.pause(0.01)
            STAGE = 1
            BLUE_PIX = [45, 87, 176]
            YEL_PIX = [210, 210, 64]
            print("Episode {}:".format(episode))
            print("  completed in {} steps".format(time + 1))
            print("  total_reward was {}".format(total_reward))
            print("Weights are " + str(weights))
            ale.reset_game()
            break
plt.show()
print('success')

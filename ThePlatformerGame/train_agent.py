import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as nnf
import torch.distributions as dist

import numpy as np
import platformer_tut8 as plat

import networkModel as nm


MAX_EPOCHS = 1500 
DISCOUNT_FACTOR = 0.99
N_TRIALS = 20
REWARD_THRESHOLD = 250 
PRINT_INTERVAL = 10

LEARNING_RATE = 0.003
LEARNING_RATE_BOOST = 0.006
MAX_BOOST_EPOCH = 800



def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    total_rewards = 0

    for reward in reversed(rewards):
        total_rewards = reward + total_rewards * discount_factor
        returns.insert(0, total_rewards)
    returns = torch.tensor(returns)
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns


def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0

    policy.train()
    observation, info = env.reset()
    prev_observation = observation

    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        distribution = dist.Categorical(policy(observation))
        action = distribution.sample()
        log_prob_action = distribution.log_prob(action)

        observation, reward, terminated, info = env.step(action.item())
        done = terminated

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward

    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)

    return episode_return, stepwise_returns, log_prob_actions


def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss


def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


#################################################################################
#                                   A* Search
#################################################################################

def manhatten(playerPosition, coinPosition):
    
    
    playerX, playerY = playerPosition
    coinX, coinY = coinPosition

    return abs(coinX - coinY) + abs(playerX - playerY) #the distance to the coin
 
# def findNearestCoin(playerPosition, coinPosition):
    
#     # if coinPosition == null: #if no coins are left
#     #     return 0

#     coinDistance = [manhatten(playerPosition, coinPosition)
#                     for coinPosition in coin_positions]

#     return coinDistance

#this needs to be nodes not player positions( the goal node and end node
#https://www.datacamp.com/tutorial/a-star-algorithm?dc_referrer=https%3A%2F%2Fwww.google.com%2F
def ASearch(playerPosition, coinPosition):
    
    coinPosition = closestGoalOrCoin()

    openNodes = [playerPosition]
    closeNodes = []

    #innitilising node propperties
    playerPosition.c = 0                                        #starting cost is 0 (the actual cost)
    playerPosition.h = manhatten(playerPosition, coinPosition)  #huristic, estimate to goal
    playerPosition.f = playerPosition.g + playerPosition.h      #total cost estimate
    playerPosition.parent = null 

    while openNodes != null:
    

        return path


# At the very top of your script
# import your class or define it

def main():   
    episode_returns = []

    
    env = plat.platformerEnv() 

    policy = nm.Network() 

    # policy.load_state_dict(torch.load('./policy.pt', weights_only=True))
    policy.train()

    optimizer = opt.Adam(policy.parameters(), lr = LEARNING_RATE)

    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)

        if episode_return > 0 and episode < MAX_BOOST_EPOCH:
            print(f'Boosting learning rate at episode {episode:3} with score {episode_return:5.1f}!')
            optimizer.param_groups[0]['lr'] = LEARNING_RATE_BOOST 
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)
        optimizer.param_groups[0]['lr'] = LEARNING_RATE

        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')
        
        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    # save the learned policy
    for param in policy.state_dict():
        print(param, "\t", policy.state_dict()[param].size())

    torch.save(policy.state_dict(), './policy.pt')

main()

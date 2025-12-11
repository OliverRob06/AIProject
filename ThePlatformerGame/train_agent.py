import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as nnf
import torch.distributions as dist

import numpy as np
import platformer_tut8 as plat

import networkModel as nm


MAX_EPOCHS = 2000
DISCOUNT_FACTOR = 0.99
N_TRIALS = 20
REWARD_THRESHOLD = 1500 
PRINT_INTERVAL = 10

LEARNING_RATE = 0.003 # 0.003
LEARNING_RATE_BOOST = 0.006
MAX_BOOST_EPOCH = 1200


def manualSaveCheck(policy):
    # Checks for S Keypress
    import pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            import sys
            import torch
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                 # save the learned policy
                for param in policy.state_dict():
                    print(param, "\t", policy.state_dict()[param].size())

                torch.save(policy.state_dict(), './policy.pt')
                return True  # Signal that a save occurred
    return False

def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    total_rewards = 0

    for reward in reversed(rewards):
        total_rewards = reward + total_rewards * discount_factor
        returns.insert(0, total_rewards)
    returns = torch.tensor(returns)
    eps = 1e-9 # Small number
    normalized_returns = (returns - returns.mean()) / (returns.std() + eps)
    return normalized_returns


def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    entropies = []   
    rewards = []
    done = False
    episode_return = 0

    policy.train()
    observation, info = env.reset()
    
    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        distribution = dist.Categorical(policy(observation))
        action = distribution.sample()

        # ... existing logging code ...
        log_prob_action = distribution.log_prob(action)
        log_prob_actions.append(log_prob_action)

        # 2. Calculate and store entropy for this step
        entropies.append(distribution.entropy()) 

        observation, reward, terminated, info = env.step(action.item())
        done = terminated
        rewards.append(reward)
        episode_return += reward

    log_prob_actions = torch.cat(log_prob_actions)
    
    # 3. Stack the entropies into a single Tensor
    entropies = torch.stack(entropies) 
    
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)

    # 4. Return entropies
    return episode_return, stepwise_returns, log_prob_actions, entropies

def calculate_loss(stepwise_returns, log_prob_actions, entropies): 
    # Standard Policy Gradient Loss
    loss = -(stepwise_returns * log_prob_actions).sum()
    
    # Entropy Bonus (Subtracts from loss to encourage exploration)
    entropy_bonus = -0.01 * entropies.sum()
    
    return loss + entropy_bonus


def update_policy(stepwise_returns, log_prob_actions, entropies, optimizer):
    stepwise_returns = stepwise_returns.detach()
    
    # Pass the entropies to the loss calculation
    loss = calculate_loss(stepwise_returns, log_prob_actions, entropies)
    
     # Print Debug Stats every 10 updates
    if np.random.rand() < 0.1:
        print(f"Loss: {loss.item():.4f} | Returns Max: {stepwise_returns.max():.2f} | Returns Min: {stepwise_returns.min():.2f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def agentStart():   

    env = plat.platformerEnv() 

    policy = nm.Network() 

    
    #policy.load_state_dict(torch.load('./policy.pt', weights_only=True))
    policy.train()
    episode_returns = []
    episode_return, stepwise_returns, log_prob_actions, entropies = forward_pass(env, policy, DISCOUNT_FACTOR)

    optimizer = opt.Adam(policy.parameters(), lr = LEARNING_RATE)

    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions, entropies = forward_pass(env, policy, DISCOUNT_FACTOR)
        manualSaveCheck(policy)
        if episode_return > 0 and episode < MAX_BOOST_EPOCH:
            print(f'Boosting learning rate at episode {episode:3} with score {episode_return:5.1f}!')
            optimizer.param_groups[0]['lr'] = LEARNING_RATE_BOOST 
        _ = update_policy(stepwise_returns, log_prob_actions, entropies, optimizer)
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

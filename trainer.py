import os
import random
import time
import torch
from torch import distributions, optim
import numpy as np
import simulation_run as run
import utils
import nn_utils as nnut
from net import Net

# Global settings
EASY = {'STATE_DIM': 66,'ACTION_DIM': 66,
        'LOW': {'BASELINE': 7.814, 'STD': 1.299}
        }
MEDIUM = {'NUM_HOST': 66, 'STATE_DIM': 132, 'ACTION_DIM': 132,
          'LOW': {'BASELINE': 7.668, 'STD': 1.259}
          }
HARD = {'NUM_HOST_HYDROXYL': 66, 'STATE_DIM_HYDROXYL': 132, 'NUM_HOST_EPOXIDE': 88, 'STATE_DIM_EPOXIDE': 176,
        'STATE_DIM': 308, 'ACTION_H_DIM': 133, 'ACTION_E_DIM': 177,
        'LOW': {'BASELINE': 7.086, 'STD': 1.308}
        }
DIFFICULTIES = (EASY, MEDIUM, HARD)


'''
Directories: 
./md/lammps_files
./DATA/{difficulty}/{exp_name}/log_and_description
./DATA/{difficulty}/{exp_name}/{simulation_name}/simulation_files
./DATA/{difficulty}/random/random.csv
./DATA/{difficulty}/FIGURES/
'''


def run_loop(difficulty: str, is_random=False, n_hydroxyl=None, n_epoxide=None,
             hidden_h_params=None, hidden_e_params=None, lr_init=None, seed=None, n_iter=2000,
             scheduled_lr=False, scheduler_period=500, log_freq=1, print_freq=5, flush_freq=10):

    # set timer and seed
    start_time = time.time()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_easy = difficulty == "easy"
    is_medium = difficulty == "medium"
    is_hard = difficulty == "hard"

    # Unpack difficulty setting
    action_dim, action_h_dim, action_e_dim = None, None, None
    state_dim_hydroxyl, state_dim_epoxide = None, None
    num_host, num_host_hydroxyl, num_host_epoxide = None, None, None

    if is_easy:
        settings = DIFFICULTIES[0]
        state_dim  = settings['STATE_DIM']
        action_dim = settings['ACTION_DIM']
    elif is_medium:
        settings = DIFFICULTIES[1]
        num_host = settings['NUM_HOST']
        state_dim = settings['STATE_DIM']
        action_dim = settings['ACTION_DIM']
    else:
        settings = DIFFICULTIES[2]
        num_host_hydroxyl = settings['NUM_HOST_HYDROXYL']
        state_dim_hydroxyl = settings['STATE_DIM_HYDROXYL']
        num_host_epoxide = settings['NUM_HOST_EPOXIDE']
        state_dim_epoxide = settings['STATE_DIM_EPOXIDE']
        state_dim = settings['STATE_DIM']
        action_h_dim = settings['ACTION_H_DIM']
        action_e_dim = settings['ACTION_E_DIM']

    baseline = settings['BASELINE']
    std = settings['STD']

    # dump results
    exp_path = f"DATA/{difficulty}"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if not is_random:
        exp_name = utils.name_experiment(difficulty, n_hydroxyl, n_epoxide, hidden_h_params, hidden_e_params,
                                     lr_init, seed, scheduled_lr, scheduler_period)
        utils.write_description(difficulty, exp_name, n_hydroxyl, n_epoxide, hidden_h_params, hidden_e_params,
                                lr_init, seed, scheduled_lr, scheduler_period)
    else:
        exp_name = "random"
        os.makedirs(f"{exp_path}/{exp_name}")
    file_log = open(f'{exp_path}/{exp_name}/{exp_name}.csv', 'w')
    file_log.write('Trajectory,Reward\n')

    # Declare networks and optimizers
    net, h_net, e_net = None, None, None
    optimizer, h_net_optimizer, e_net_optimizer = None, None, None
    scheduler, h_net_scheduler, e_net_scheduler = None, None, None

    lr_func = lambda itr: 0.5 ** itr
    if is_easy or is_medium:
        net = Net(input_size=state_dim, params=hidden_h_params, output_size=action_dim)
        optimizer = optim.Adam(net.parameters(), lr=lr_init)

        if scheduled_lr:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        h_net = Net(input_size=state_dim, params=hidden_h_params, output_size=action_h_dim)
        h_net_optimizer = optim.Adam(h_net.parameters(), lr=lr_init)
        e_net = Net(input_size=state_dim, params=hidden_e_params, output_size=action_e_dim)
        e_net_optimizer = optim.Adam(e_net.parameters(), lr=lr_init)
        if scheduled_lr:
            h_net_scheduler = optim.lr_scheduler.LambdaLR(h_net_optimizer, lr_func)
            e_net_scheduler = optim.lr_scheduler.LambdaLR(e_net_optimizer, lr_func)

    # ***********************************************************************
    # MAIN TRAINING PROCESS *************************************************
    # ***********************************************************************

    rewards_all_iter = []  # tracks final rewards for all iterations

    # Iterations start here
    if is_easy or is_medium:
        locations_final = []
        for iteration in range(n_iter):
            print("Current learning rate:", optimizer.param_groups[0]['lr'])
            state = [0] * state_dim
            state_tensor = torch.tensor(state, dtype=torch.float)
            locations = []
            list_of_action_and_action_dist = []
            reward = None
            for j in range(n_hydroxyl):
                action_probs = net(state_tensor) if not is_random else torch.rand(action_dim)
                if is_medium:
                    for location in locations:
                        if location < num_host:
                            action_probs[[location, location + num_host]] = 0
                        else:
                            action_probs[[location, location - num_host]] = 0
                else:
                    action_probs[locations] = 0
                action_distribution = distributions.Categorical(probs=action_probs)
                action = action_distribution.sample()
                list_of_action_and_action_dist.append((action, action_distribution))
                action_one_hot = [0] * state_dim
                action_one_hot[action.item()] = 1
                state = [a + b for a, b in zip(state, action_one_hot)]
                state_tensor = torch.tensor(state, dtype=torch.float)
                locations = utils.state2loc(state)
                #  compute the final reward
                if j == n_hydroxyl - 1:
                    # if location is the same as previous, use previous reward immediately
                    if set(locations) == set(locations_final):
                        reward = rewards_all_iter[-1]
                        print(f'At iteration {iteration} location not changed, use previous reward')
                    # else, run simulation and compute the reward
                    else:
                        toughness, _ = run.run_go_h(difficulty, exp_name, locations, iteration)
                        reward = (toughness - baseline) / std
                    locations_final = locations.copy()
            rewards_all_iter.append(reward)
            # Neural net update
            if not is_random:
                loss = nnut.calc_loss(list_of_action_and_action_dist, reward)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduled_lr:
                    if iteration % scheduler_period == 0 and iteration != 0  and iteration < 2000:
                        scheduler.step()

            utils.print_log(file_log, iteration, reward, print_freq, log_freq, flush_freq)

    else:
        locations_hydroxyl_final = []
        locations_epoxide_final = []
        for iteration in range(n_iter):
            m, n = n_hydroxyl, n_epoxide
            state = [0] * state_dim
            state_tensor = torch.tensor(state, dtype=torch.float)
            list_of_action_and_action_dist_h = []
            list_of_action_and_action_dist_e = []
            final_net = None
            # Assignments start here
            while m + n > 0:
                # randomly pick a net
                net_index, m, n, func_net, optimizer, scheduler = nnut.pick_net(m, n, h_net, e_net,
                    h_net_optimizer, e_net_optimizer, scheduled_lr, h_net_scheduler, e_net_scheduler)
                if m + n == 0:
                    final_net = net_index
                # assign one functional group using func_net
                if not is_random:
                    action_probs = func_net(state_tensor)
                else:
                    action_probs = torch.rand(action_h_dim) if net_index==1 else torch.rand(action_e_dim)
                action_probs_feasible = utils.zero_infeasible(state, action_probs, is_hydroxyl=net_index==1)
                action_distribution = distributions.Categorical(probs=action_probs_feasible)
                action = action_distribution.sample()

                if net_index == 1:
                    list_of_action_and_action_dist_h.append((action, action_distribution))
                    action_one_hot = torch.zeros(action_h_dim)
                    action_one_hot[action.item()] = 1
                    placeholder = torch.zeros(state_dim_epoxide)
                    action_one_hot_complete = torch.cat((action_one_hot[:-1], placeholder), dim=-1)
                else:
                    list_of_action_and_action_dist_e.append((action, action_distribution))
                    action_one_hot = torch.zeros(action_e_dim)
                    action_one_hot[action.item()] = 1
                    placeholder = torch.zeros(state_dim_hydroxyl)
                    action_one_hot_complete = torch.cat((placeholder, action_one_hot[:-1]), dim=-1)
                state = [a + b for a, b in zip(state, action_one_hot_complete)]
                state_tensor = torch.tensor(state, dtype=torch.float)
            state_hydroxyl, state_epoxide = state[:state_dim_hydroxyl], state[state_dim_hydroxyl:]

            # collect reward and update net parameters
            locations_hydroxyl = utils.state2loc(state_hydroxyl)
            locations_epoxide = utils.state2loc(state_epoxide)
            # if location is the same as previous, use previous reward immediately
            if set(locations_hydroxyl) == set(locations_hydroxyl_final) and \
                    set(locations_epoxide) == set(locations_epoxide_final):
                reward = rewards_all_iter[-1]
                print(f'At iteration {iteration} location not changed, use previous reward')
            # else, run simulation and compute the reward
            else:
                toughness, _ = run.run_go_he(difficulty, exp_name, locations_hydroxyl, locations_epoxide, iteration)
                reward = toughness  # (toughness - baseline) / std
            locations_hydroxyl_final = locations_hydroxyl.copy()
            locations_epoxide_final = locations_epoxide.copy()
            rewards_all_iter.append(reward)

            # Neural net update
            if not is_random:
                if final_net == 1:
                    loss = nnut.calc_loss(list_of_action_and_action_dist_h, reward)
                else:
                    loss = nnut.calc_loss(list_of_action_and_action_dist_e, reward)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduled_lr:
                    if iteration % scheduler_period == 0 and iteration != 0 and iteration < 2000:
                        scheduler.step()

            utils.print_log(file_log, iteration, reward, print_freq, log_freq, flush_freq)

    file_log.close()
    print(f'Execution time: {((time.time() - start_time) / 3600):.3f} hours')

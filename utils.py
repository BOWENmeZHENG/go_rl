import numpy as np
import os

from data_process import read
import shutil
import pickle

# Global settings
num_host_hydroxyl = 66
num_state_hydroxyl = num_host_hydroxyl * 2
num_host_epoxide = 88
num_state_epoxide = num_host_epoxide * 2
state_dim = num_state_hydroxyl + num_state_epoxide
action_h_dim = num_state_hydroxyl + 1
action_e_dim = num_state_epoxide + 1


def loc2state(locations: list):
    state = [0] * 66
    for i in locations:
        state[i] = 1
    return state


def state2loc(state) -> list:
    locations = []
    for i, entry in enumerate(state):
        if entry == 1:
            locations.append(i)
    return locations


def read_curve(curve: str):
    with open(f'{curve}', 'r') as file_curve:
        raw_data = file_curve.read().splitlines()
        raw_data.pop(0)
    raw_data_length = len(raw_data)
    strain_array, stress_array = np.zeros((raw_data_length, 1)), np.zeros((raw_data_length, 1))
    i_zero = None
    for i, datapoint in enumerate(raw_data):
        strain_stress = datapoint.split()
        strain_array[i], stress_array[i] = float(strain_stress[0]), float(strain_stress[1])
        if stress_array[i] <= 0 and i > 100:
            i_zero = i
            break
    energy = np.trapz(stress_array[:i_zero].T, strain_array[:i_zero].T)[0]
    return strain_array[:i_zero], stress_array[:i_zero], energy


def old_data():
    locations, energies = [], []
    exp_names = sorted([filename for filename in os.listdir("data") if filename.startswith("exp")])
    for experiment in exp_names:
        directory = f"data/{experiment}"
        ds_names = sorted([filename for filename in os.listdir(directory) if filename.startswith("dataset")])
        for dataset in ds_names:
            directory = f"data/{experiment}/{dataset}"
            states_ds, energies_ds = read(directory)
            locations.extend([state2loc(state) for state in states_ds])
            energies.extend(energies_ds)
    return locations, energies


def moving_average(rewards, window: int):
    return rewards.rolling(window=window, min_periods=1).mean()


def curves_mean_std(results: dict, keys: list, length: int):
    curve_mean, curve_low, curve_high = [], [], []
    for i in range(length):
        data_iter = []
        for key in keys:
            data_iter.append(results[key]["Reward"][i])
        data_iter_np = np.array(data_iter)
        data_iter_mean, data_iter_std = np.mean(data_iter_np), np.std(data_iter_np)
        curve_mean.append(data_iter_mean)
        curve_low.append(data_iter_mean - data_iter_std)
        curve_high.append(data_iter_mean + data_iter_std)
    return curve_mean, curve_low, curve_high


def copy_code_set(names: list):
    if not os.path.exists("EXP"):
        os.mkdir("EXP")
    for name in names:
        exp_dir = f"EXP/{name}"
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        py_files = [f for f in os.listdir() if f.endswith('.py')]
        ipynb_files = [f for f in os.listdir() if f.endswith('.ipynb')]
        folder = "md"
        for py_file in py_files:
            shutil.copyfile(py_file, f'{exp_dir}/{py_file}')
        for ipynb_file in ipynb_files:
            shutil.copyfile(ipynb_file, f'{exp_dir}/{ipynb_file}')
        shutil.copytree(folder, f'{exp_dir}/{folder}')


def name_experiment(difficulty: str, n_hydroxyl=0, n_epoxide=0,
                    hidden_h_params=None, hidden_e_params=None,
                    lr_init=None, seed=None,
                    scheduled_lr=False, scheduler_period=None):
    exp_path = f"DATA/{difficulty}"
    if scheduled_lr:
        exp_name = f"hn_{n_hydroxyl}_en_{n_epoxide}_hp_{hidden_h_params}_ep_{hidden_e_params}" \
               f"_l_{lr_init}_s_{seed}_p_{scheduler_period}"
    else:
        exp_name = f"hn_{n_hydroxyl}_en_{n_epoxide}_hp_{hidden_h_params}_ep_{hidden_e_params}_l_{lr_init}_s_{seed}"
    os.makedirs(f"{exp_path}/{exp_name}")
    return exp_name


def write_description(difficulty: str, exp_name: str, n_hydroxyl=0, n_epoxide=0,
                      hidden_h_params=None, hidden_e_params=None,
                      lr_init=None, seed=None, scheduled_lr=False, scheduler_period=None):
    exp_path = f"DATA/{difficulty}"
    with open(f'{exp_path}/{exp_name}/{exp_name}.description', 'w') as file_description:
        file_description.write(f'Name: {exp_name}\n')
        file_description.write(f'Number of hydroxyl groups: {n_hydroxyl}\n')
        file_description.write(f'Number of epoxide groups: {n_epoxide}\n')
        file_description.write(f'H_Net hidden layer parameters: {hidden_h_params}\n')
        file_description.write(f'E_Net hidden layer parameters: {hidden_e_params}\n')
        file_description.write(f'Initial learning Rate: {lr_init}\n')
        file_description.write(f'Seed: {seed}\n')
        file_description.write(f'Learning rate scheduler: {scheduled_lr}\n')
        if scheduled_lr:
            file_description.write(f'Scheduler period: {scheduler_period}\n')


def zero_infeasible(state, action_probs, is_hydroxyl=True):
    # take in state to eliminate infeasible actions

    # load a map going from the index of epoxide group to the two indices of atoms
    with open("md/pairs_indices", "rb") as fp:
        mapping = pickle.load(fp)  # list of tuples

    # partition state
    s_h_upper = state[:num_host_hydroxyl]
    s_h_lower = state[num_host_hydroxyl:num_host_hydroxyl * 2]
    s_e_upper = state[num_host_hydroxyl * 2:num_host_hydroxyl * 2 + num_host_epoxide]
    s_e_lower = state[num_host_hydroxyl * 2 + num_host_epoxide:]

    if is_hydroxyl:  # action is to assign a hydroxyl group

        # set infeasible actions zero probability according to current hydroxyl distribution
        indices_h_upper = [i for i, state_entry in enumerate(s_h_upper) if state_entry == 1]
        for index in indices_h_upper:
            action_probs[index], action_probs[index + num_host_hydroxyl] = 0, 0
        indices_h_lower = [i for i, state_entry in enumerate(s_h_lower) if state_entry == 1]
        for index in indices_h_lower:
            action_probs[index], action_probs[index + num_host_hydroxyl] = 0, 0

        # set infeasible actions zero probability according to current epoxide distribution
        atom_indices_so_far = set()
        # need pair_index and atom_indices
        pair_indices_e_upper = [i for i, state_entry in enumerate(s_e_upper) if state_entry == 1]
        for pair_index in pair_indices_e_upper:
            two_atoms_indices = mapping[pair_index]
            atom_indices_so_far.add(two_atoms_indices[0])
            atom_indices_so_far.add(two_atoms_indices[1])
        pair_indices_e_lower = [i for i, state_entry in enumerate(s_e_lower) if state_entry == 1]
        for pair_index in pair_indices_e_lower:
            two_atoms_indices = mapping[pair_index]
            atom_indices_so_far.add(two_atoms_indices[0])
            atom_indices_so_far.add(two_atoms_indices[1])
        atom_indices_all = list(atom_indices_so_far)
        # set infeasible actions zero probability
        for index in atom_indices_all:
            action_probs[index], action_probs[index + num_host_hydroxyl] = 0, 0

    else:  # action is to assign an epoxide group
        atoms_indices_so_far = set()
        # set infeasible actions zero probability according to current epoxide distribution
        indices_e_upper = [i for i, state_entry in enumerate(s_e_upper) if state_entry == 1]
        # generate atom indices from pair indices
        for index in indices_e_upper:
            atom_pair = mapping[index]
            atoms_indices_so_far.add(atom_pair[0])
            atoms_indices_so_far.add(atom_pair[1])
        indices_e_lower = [i for i, state_entry in enumerate(s_e_lower) if state_entry == 1]
        for index in indices_e_lower:
            atom_pair = mapping[index]
            atoms_indices_so_far.add(atom_pair[0])
            atoms_indices_so_far.add(atom_pair[1])

        # zero illegal actions
        atoms_indices_all = list(atoms_indices_so_far)

        indices_to_zero = [i for i, pair in enumerate(mapping)
                           if pair[0] in atoms_indices_all or pair[1] in atoms_indices_all]

        for index in indices_to_zero:
            action_probs[index], action_probs[index + num_host_epoxide] = 0, 0

        # set infeasible actions zero probability according to current hydroxyl distribution
        pair_indices_so_far = set()

        atoms_h_upper = [i for i, state_entry in enumerate(s_h_upper) if state_entry == 1]
        for atom in atoms_h_upper:
            pair_indices = [i for i, pair in enumerate(mapping) if pair[0] == atom or pair[1] == atom]
            for pair_index in pair_indices:
                pair_indices_so_far.add(pair_index)
        atoms_h_lower = [i for i, state_entry in enumerate(s_h_lower) if state_entry == 1]
        for atom in atoms_h_lower:
            pair_indices = [i for i, pair in enumerate(mapping) if pair[0] == atom or pair[1] == atom]
            for pair_index in pair_indices:
                pair_indices_so_far.add(pair_index)
        pair_indices_all = list(pair_indices_so_far)
        # set infeasible actions zero probability
        for index in pair_indices_all:
            action_probs[index], action_probs[index + num_host_epoxide] = 0, 0

    return action_probs


def print_log(file_log, iteration, reward, print_freq, log_freq, flush_freq):
    if iteration % print_freq == 0:
        print(iteration, reward)
    if iteration % log_freq == 0:
        file_log.write(f'{iteration},{reward}\n')
    if iteration % flush_freq == 0:
        file_log.flush()


def append_early_stop(difficulty: str, exp_name: str, target_iter=1999):
    log_path = f'DATA/{difficulty}/{exp_name}/{exp_name}.csv'
    with open(log_path, "r") as f:
        for line in f:
            pass
        last_line = line
        last_iteration = last_line.split(",")[0]
        last_reward = last_line.split(",")[1]

    with open(log_path, "a") as f:
        for iteration in range(int(last_iteration), target_iter):
            iteration += 1
            f.write(f"{iteration},{last_reward}")


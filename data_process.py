import os
import utils


def read(directory: str):
    states, energies = [], []
    for traj_index in sorted(os.listdir(directory)):
        os.chdir(directory + "/" + traj_index)

        for traj_name in sorted(os.listdir()):
            if any(map(str.isdigit, traj_name)):
                # Read locations to states
                lis = list(map(int, traj_name.split("_")[1:-1]))
                state = utils.loc2state(lis)
                states.append(state)
                # Read energies
                os.chdir(traj_name)
                curve_filename = [filename for filename in os.listdir() if filename.startswith("curve")][0]
                _, _, energy = utils.read_curve(curve_filename)
                energies.append(energy)
                os.chdir("..")
        os.chdir('../../../..')
    return states, energies


def read_all(dir_list):
    states_all, energies_all = [], []
    for directory in dir_list:
        states, energies = read(directory)
        states_all.extend(states)
        energies_all.extend(energies)
    return states_all, energies_all





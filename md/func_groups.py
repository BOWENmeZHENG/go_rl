import numpy as np
from scipy.spatial import distance
import pickle

BL = 1.421


def epoxide(atoms_relax, locations: list):  # 88 possible COC locations, range of location indices: 0 to 175
    # get all potential pairs
    pairs_indices, pairs_coordinates, pairs_center = [], [], []
    num_atoms_relax = np.shape(atoms_relax)[0]
    for atom in range(num_atoms_relax):
        for another_atom in range(atom + 1, num_atoms_relax):
            dist = distance.euclidean(atoms_relax[atom], atoms_relax[another_atom])
            if dist < BL * 1.2:
                pair_indices = atom, another_atom
                pair_coordinates = atoms_relax[pair_indices[0]], atoms_relax[pair_indices[1]]
                pair_center = np.mean(pair_coordinates, axis=0)
                pairs_indices.append(pair_indices)
                pairs_coordinates.append(pair_coordinates)
                pairs_center.append(pair_center)

    pairs_coordinates = np.array(pairs_coordinates)
    pairs_center = np.array(pairs_center)
    '''
    atoms_o: np.array, coordinates of O atoms, shape = (len(locations), 3)
    pairs_indices: list of tuples, indices of carbon atom pairs that are involved in C-O-C, length = 88
    pairs_coordinates: np.array, coordinates of carbon atoms that are involved in C-O-C, shape = (88, 2, 3)
    pairs_center: np.array, midpoint coordinate of carbon atoms that are involved in C-O-C, shape = (88, 3)
    pairs_indices_full: list of tuples, pairs_indices considering both sides, length = 176
    pairs_coordinates_full: np.array, pairs_coordinates considering both sides, shape = (176, 2, 3)
    pairs_center_full: np.array, pairs_center considering both sides, shape = (176, 3)
    '''
    # Assign oxygen atoms
    num_coc = np.shape(pairs_center)[0]
    atoms_coc_up, atoms_coc_down = [], []
    # Determine which atoms to add functional groups to
    for location in locations:
        if location < num_coc:  # top side
            atoms_coc_up.append(pairs_center[location])
        else:  # bottom side
            atoms_coc_down.append(pairs_center[location - num_coc])
    atoms_o_up, atoms_o_down, = np.array(atoms_coc_up.copy()), np.array(atoms_coc_down.copy())
    if atoms_o_up.size == 0:
        atoms_o_down[:, 2] -= 1.40
        atoms_coc = atoms_o_down
    elif atoms_o_down.size == 0:
        atoms_o_up[:, 2] += 1.40
        atoms_coc = atoms_o_up
    else:
        atoms_o_up[:, 2] += 1.40
        atoms_o_down[:, 2] -= 1.40
        atoms_coc = np.vstack((atoms_o_up, atoms_o_down))

    # write pairs_indices to file
    with open("md/pairs_indices", "wb") as fp:
        pickle.dump(pairs_indices, fp)

    return atoms_coc, pairs_indices, pairs_coordinates, pairs_center


def hydroxyl_2_sides(atoms_relax, locations: list):  # 66 possible COH locations, range of location indices: 0 to 131
    # Initialization
    atoms_coh_up, atoms_coh_down = [], []
    num_atoms_relax = np.shape(atoms_relax)[0]
    # Determine which atoms to add functional groups to
    for location in locations:
        if location < num_atoms_relax:  # top side
            atoms_coh_up.append(atoms_relax[location])
        else:  # bottom side
            atoms_coh_down.append(atoms_relax[location - num_atoms_relax])
    # Define functional groups
    atoms_coh_o_up, atoms_coh_h_up, = np.array(atoms_coh_up.copy()), np.array(atoms_coh_up.copy())
    atoms_coh_o_down, atoms_coh_h_down = np.array(atoms_coh_down.copy()), np.array(atoms_coh_down.copy())
    if atoms_coh_o_up.size == 0:
        atoms_coh_o_down[:, 2] -= 1.50
        atoms_coh_h_down[:, 2] -= 2.61
        atoms_coh_o, atoms_coh_h = atoms_coh_o_down, atoms_coh_h_down
    elif atoms_coh_o_down.size == 0:
        atoms_coh_o_up[:, 2] += 1.50
        atoms_coh_h_up[:, 2] += 2.61
        atoms_coh_o, atoms_coh_h = atoms_coh_o_up, atoms_coh_h_up
    else:
        atoms_coh_o_up[:, 2] += 1.50
        atoms_coh_h_up[:, 2] += 2.61
        atoms_coh_o_down[:, 2] -= 1.50
        atoms_coh_h_down[:, 2] -= 2.61
        atoms_coh_o = np.vstack((atoms_coh_o_up, atoms_coh_o_down))
        atoms_coh_h = np.vstack((atoms_coh_h_up, atoms_coh_h_down))
    return atoms_coh_o, atoms_coh_h


def hydroxyl_1_side(atoms_relax, locations: list):
    atoms_coh = atoms_relax[locations, :]
    atoms_coh_o = atoms_coh.copy()
    atoms_coh_h = atoms_coh.copy()
    for i, _ in enumerate(locations):
        atoms_coh_o[i, 2] = 1.50
        atoms_coh_h[i, 2] = 2.61
    return atoms_coh_o, atoms_coh_h




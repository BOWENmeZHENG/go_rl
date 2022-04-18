import numpy as np
import os
from shutil import copyfile

BL = 1.421


def basal_plane() -> tuple:
    """
    Define graphene basal plane for MD simulation.
    """
    n = 7
    m = 5
    total_atom = 2 + 4 * n + (2 * n + 2) * (m - 1)
    atom_coordinate = np.zeros((total_atom, 3))
    num_edge_atom = 2 * m
    for i in range(num_edge_atom):
        r = (i + 1) % 4
        if r == 1:
            # atom_coordinate[i, 0] = 0
            atom_coordinate[i, 1] = 1 / 2 * BL + ((i + 1) - r) / 4 * (3 * BL)
        elif r == 2:
            # atom_coordinate[i, 0] = 0
            atom_coordinate[i, 1] = 3 / 2 * BL + ((i + 1) - r) / 4 * (3 * BL)
        elif r == 3:
            atom_coordinate[i, 0] = (n + 1 / 2) * np.sqrt(3) * BL
            atom_coordinate[i, 1] = 2 * BL + ((i + 1) - r) / 4 * (3 * BL)
        else:
            atom_coordinate[i, 0] = (n + 1 / 2) * np.sqrt(3) * BL
            atom_coordinate[i, 1] = 3 * BL + (((i + 1) - r) / 4 - 1) * (3 * BL)

    del r
    j = 1
    for i in range(num_edge_atom, total_atom):
        new_num = i + 1 - num_edge_atom - 4 * n * (j - 1)
        r = new_num % 4
        if r == 1:
            atom_coordinate[i, 0] = (np.sqrt(3) / 2 + np.sqrt(3) * (new_num - r) / 4) * BL
            atom_coordinate[i, 1] = 2 * BL + 3 * BL * (j - 1)
        elif r == 2:
            atom_coordinate[i, 0] = (np.sqrt(3) + np.sqrt(3) * (new_num - r) / 4) * BL
            atom_coordinate[i, 1] = 3 / 2 * BL + 3 * BL * (j - 1)
        elif r == 3:
            atom_coordinate[i, 0] = (np.sqrt(3) + np.sqrt(3) * (new_num - r) / 4) * BL
            atom_coordinate[i, 1] = 1 / 2 * BL + 3 * BL * (j - 1)
        else:
            atom_coordinate[i, 0] = (np.sqrt(3) / 2 + np.sqrt(3) * ((new_num - r) / 4 - 1)) * BL
            atom_coordinate[i, 1] = 3 * BL * (j - 1)
        if (new_num - r) / 4 >= n:
            j += 1

    # xmin = min(atom_coordinate[:, 0])
    xmax = max(atom_coordinate[:, 0])
    # ymin = min(atom_coordinate[:, 1])
    # ymax = max(atom_coordinate[:, 1])

    atoms_fixed, atoms_move, atoms_relax = [], [], []
    clamp_left, clamp_right = 2, 2.5
    for i in range(total_atom):
        if atom_coordinate[i, 0] < clamp_left:
            atoms_fixed.append(atom_coordinate[i])
        elif atom_coordinate[i, 0] > xmax - clamp_right:
            atoms_move.append(atom_coordinate[i])
        else:
            atoms_relax.append(atom_coordinate[i])
    return np.array(atoms_fixed), np.array(atoms_move), np.array(atoms_relax)


def assemble(atoms_fixed, atoms_move, atoms_relax, atoms_coh_o, atoms_coh_h, atoms_coc=None):
    """
    Assemble GBP and functional group atoms for MD simulation
    :param atoms_fixed: fixed GBP atoms
    :param atoms_move: GBP atoms for loading
    :param atoms_relax: GBP atoms that host functional groups
    :param atoms_coh_o: Oxygen atoms for hydroxyl groups
    :param atoms_coh_h: Hydrogen atoms for hydroxyl groups
    :param atoms_coc: Oxygen atoms for epoxide groups
    :return: atoms_xyz
    """
    num_atoms_fixed = np.shape(atoms_fixed)[0]
    num_atoms_move = np.shape(atoms_move)[0]
    num_atoms_relax = np.shape(atoms_relax)[0]
    num_atoms_coh = np.shape(atoms_coh_o)[0]
    atoms_fixed = np.hstack((2 * np.ones((num_atoms_fixed, 1)), np.ones((num_atoms_fixed, 1)), atoms_fixed))
    atoms_move = np.hstack((3 * np.ones((num_atoms_move, 1)), np.ones((num_atoms_move, 1)), atoms_move))
    atoms_relax = np.hstack((np.ones((num_atoms_relax, 1)), np.ones((num_atoms_relax, 1)), atoms_relax))
    atoms_coh_o = np.hstack((np.ones((num_atoms_coh, 1)), 3 * np.ones((num_atoms_coh, 1)), atoms_coh_o))
    atoms_coh_h = np.hstack((np.ones((num_atoms_coh, 1)), 2 * np.ones((num_atoms_coh, 1)), atoms_coh_h))
    if atoms_coc is None:
        atoms_xyz = np.vstack((atoms_fixed, atoms_move, atoms_relax, atoms_coh_o, atoms_coh_h))
    else:
        num_atoms_coc = np.shape(atoms_coc)[0]
        atoms_coc = np.hstack((np.ones((num_atoms_coc, 1)), 3 * np.ones((num_atoms_coc, 1)), atoms_coc))
        atoms_xyz = np.vstack((atoms_fixed, atoms_move, atoms_relax, atoms_coc, atoms_coh_o, atoms_coh_h))
    return atoms_xyz

def assemble_graphene(atoms_fixed, atoms_move, atoms_relax):
    """
    Assemble atoms for empty graphene
    :param atoms_fixed: fixed atoms
    :param atoms_move: atoms for loading
    :param atoms_relax: atoms that relax
    :return: atoms_xyz
    """
    num_atoms_fixed = np.shape(atoms_fixed)[0]
    num_atoms_move = np.shape(atoms_move)[0]
    num_atoms_relax = np.shape(atoms_relax)[0]
    atoms_fixed = np.hstack((2 * np.ones((num_atoms_fixed, 1)), np.ones((num_atoms_fixed, 1)), atoms_fixed))
    atoms_move = np.hstack((3 * np.ones((num_atoms_move, 1)), np.ones((num_atoms_move, 1)), atoms_move))
    atoms_relax = np.hstack((np.ones((num_atoms_relax, 1)), np.ones((num_atoms_relax, 1)), atoms_relax))
    atoms_xyz = np.vstack((atoms_fixed, atoms_move, atoms_relax))
    return atoms_xyz


def write_data(difficulty: str, exp_name: str, simulation_name: str, atoms_xyz, is_graphene_oxide=True):
    """
    Write .data file for MD simulation
    """
    xlo = min(atoms_xyz[:, 2]) - 20
    xhi = max(atoms_xyz[:, 2]) + 20
    ylo = min(atoms_xyz[:, 3]) - 5
    yhi = max(atoms_xyz[:, 3]) + 5
    zlo = min(atoms_xyz[:, 4]) - 5
    zhi = max(atoms_xyz[:, 4]) + 5
    num_atoms = np.shape(atoms_xyz)[0]
    simulation_path = f'DATA/{difficulty}/{exp_name}'
    if not os.path.exists(f'{simulation_path}/{simulation_name}'):
        os.makedirs(f'{simulation_path}/{simulation_name}')
    with open(f'{simulation_path}/{simulation_name}/{simulation_name}.data', 'w') as file_data:
        file_data.write('# \n\n')
        file_data.write(f'{num_atoms} atoms\n')
        file_data.write('0 bonds\n')
        file_data.write('0 angles\n')
        file_data.write('0 dihedrals\n')
        file_data.write('\n')
        file_data.write('3 atom types\n') if is_graphene_oxide else file_data.write('1 atom types\n')
        file_data.write(f'{xlo:.3f} {xhi:.3f} xlo xhi\n')
        file_data.write(f'{ylo:.3f} {yhi:.3f} ylo yhi\n')
        file_data.write(f'{zlo:.3f} {zhi:.3f} zlo zhi\n')
        file_data.write('\n\n')
        file_data.write('Masses\n\n')
        file_data.write('1   12.0107\n')
        if is_graphene_oxide:
            file_data.write('2   1.00784\n')
            file_data.write('3   15.999\n')
        file_data.write('\n')
        file_data.write('Atoms\n\n')
        for i in range(num_atoms):
            file_data.write(f'{i + 1} {int(atoms_xyz[i, 0])} {int(atoms_xyz[i, 1])} {0:.5f} '
                            f'{atoms_xyz[i, 2]:.5f} {atoms_xyz[i, 3]:.5f} {atoms_xyz[i, 4]:.5f}\n')


def write_in(difficulty: str, exp_name: str, simulation_name: str, atoms_xyz, is_graphene_oxide=True):
    """
    Write .in file for MD simulation
    """
    simulation_path = f'DATA/{difficulty}/{exp_name}'
    if not os.path.exists(f'{simulation_path}/{simulation_name}'):
        os.makedirs(f'{simulation_path}/{simulation_name}')
    with open(f'{simulation_path}/{simulation_name}/in.{simulation_name}', 'w') as file_in:
        file_in.write('# \n\n')
        file_in.write('units 		real\n')
        file_in.write('timestep	    0.1\n')
        file_in.write('dimension 	3\n')
        file_in.write('boundary 	p p p\n')
        file_in.write(f'log 	{simulation_name}.log\n')
        file_in.write('atom_style 	full\n')
        file_in.write(f'read_data 	{simulation_name}.data\n')
        file_in.write('group 		FIXED molecule 2\n')
        file_in.write('group 		MOVE molecule 3\n')
        file_in.write('group 		GNR molecule 1\n')
        file_in.write('fix 		2 FIXED setforce NULL 0 0\n')
        file_in.write('fix 		3 MOVE setforce NULL 0 0\n')
        file_in.write(f'variable	LENGTH equal "{max(atoms_xyz[:, 2]) - min(atoms_xyz[:, 2]):.5f}"\n')
        file_in.write(f'variable	WIDTH equal "{max(atoms_xyz[:, 3]) - min(atoms_xyz[:, 3]):.5f}"\n')
        file_in.write('variable	THICKNESS equal "7.76"\n')
        # C. Chen, Q. H. Yang and Y. Yang, Self-assembled freestanding graphite oxide membrane,
        # Adv. Mater., 2009, 21, 3007–3018.
        file_in.write('variable	VOLUME equal v_LENGTH*v_WIDTH*v_THICKNESS*1e-30\n')
        file_in.write('compute 	MOVEdisp MOVE displace/atom\n')
        file_in.write('compute 	MOVEDISP MOVE reduce ave c_MOVEdisp[1]\n')
        file_in.write('variable 	STRAIN equal c_MOVEDISP/v_LENGTH\n')
        file_in.write('compute 	PE all pe\n')
        file_in.write('compute 	STRESSATOM all stress/atom NULL pair\n')
        file_in.write('compute 	STRESSTEMP1 all reduce sum c_STRESSATOM[*]\n')
        file_in.write('variable 	STRESSTEMP2 equal 1e-25*c_STRESSTEMP1[1]\n')
        file_in.write('variable 	STRESS equal v_STRESSTEMP2/v_VOLUME*1e-9 # Unit: GPa\n')
        file_in.write('pair_style reax/c NULL\n')
        file_in.write('pair_coeff * * ffield.reax.cho C H O\n') if is_graphene_oxide \
            else file_in.write('pair_coeff * * ffield.reax.cho C\n')
        file_in.write('fix 0 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n')
        file_in.write('velocity 	GNR create 300 1 mom yes rot yes dist gaussian\n')
        file_in.write('fix		NPT all npt temp 300.0 300.0 1 iso 1 1 1000\n')
        file_in.write(f'dump       1 all custom 400 {simulation_name}.lammpstrj id type x y z\n')
        file_in.write('thermo 		100\n')
        file_in.write('thermo_style    custom v_STRAIN v_STRESS\n')
        file_in.write('minimize 	1.0e-4 1.0e-6 100 1000\n')
        file_in.write('thermo 		100\n')
        file_in.write('run 		5000\n')
        file_in.write('unfix 		NPT\n')
        file_in.write('unfix 		2\n')
        file_in.write('unfix 		3\n')
        file_in.write('fix 		2 FIXED setforce 0 0 0\n')
        file_in.write('fix 		3 MOVE setforce 0 0 0\n')
        file_in.write('fix		NVE all nve\n')
        file_in.write('velocity 	FIXED set 0 0 0 sum no units box\n')
        file_in.write('velocity 	MOVE set 0.01 0 0 sum no units box\n')
        file_in.write(f'fix 		PRINT all print 10 "${{STRAIN}}   ${{STRESS}}" file curve_{simulation_name}.txt\n')
        file_in.write('run 		8000\n')


def run_md(difficulty: str, exp_name: str, simulation_name: str):
    """
    run MD simulation
    """
    simulation_path = f'DATA/{difficulty}/{exp_name}'
    copyfile('md/lmp_serial', f'{simulation_path}/{simulation_name}/lmp_serial')
    copyfile('md/ffield.reax.cho', f'{simulation_path}/{simulation_name}/ffield.reax.cho')
    os.chdir(f'{simulation_path}/{simulation_name}')
    os.system('chmod u=rwx,g=rx,o=r lmp_serial')
    os.system(f'./lmp_serial -screen none -in in.{simulation_name}')
    os.chdir('../../../..')
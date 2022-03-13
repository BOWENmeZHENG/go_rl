import md.func_groups as func
import md.md_utils as mdut
import utils
import os


def run_go_he(difficulty: str, exp_name: str,
              locations_hydroxyl: list, locations_epoxide: list, iter_n: int, is_random=False):
    simulation_path = f'DATA/{difficulty}/{exp_name}'
    if is_random:
        simulation_name = 'r_'
    else:
        simulation_name = '_'
    simulation_name += 'h_'
    for location in locations_hydroxyl:
        simulation_name += str(location)
        simulation_name += '_'
    simulation_name += 'e_'
    for location in locations_epoxide:
        simulation_name += str(location)
        simulation_name += '_'
    simulation_name += f"iter_{iter_n}"
    atoms_fixed, atoms_move, atoms_relax = mdut.basal_plane()
    atoms_coc, *_ = func.epoxide(atoms_relax, locations_epoxide)
    atoms_coh_o, atoms_coh_h = func.hydroxyl_2_sides(atoms_relax, locations_hydroxyl)
    atoms_xyz = mdut.assemble(atoms_fixed, atoms_move, atoms_relax, atoms_coh_o, atoms_coh_h, atoms_coc)
    mdut.write_data(difficulty, exp_name, simulation_name, atoms_xyz)
    mdut.write_in(difficulty, exp_name, simulation_name, atoms_xyz)
    mdut.run_md(difficulty, exp_name, simulation_name)
    strain_array_go, stress_array_go, energy_go = utils.read_curve(f'{simulation_path}/{simulation_name}'
                                                                   f'/curve_{simulation_name}.txt')
    os.remove(f'{simulation_path}/{simulation_name}/lmp_serial')
    return energy_go, simulation_name


def run_go_h(difficulty: str, exp_name: str, locations: list, iter_n: int, is_random=False):
    simulation_path = f'DATA/{difficulty}/{exp_name}'
    if is_random:
        simulation_name = 'r_'
    else:
        simulation_name = '_'
    for location in locations:
        simulation_name += str(location)
        simulation_name += '_'
    simulation_name += f"iter_{iter_n}"
    atoms_fixed, atoms_move, atoms_relax = mdut.basal_plane()
    if difficulty == "medium":
        atoms_coh_o, atoms_coh_h = func.hydroxyl_2_sides(atoms_relax, locations)
    else:
        atoms_coh_o, atoms_coh_h = func.hydroxyl_1_side(atoms_relax, locations)
    atoms_xyz = mdut.assemble(atoms_fixed, atoms_move, atoms_relax, atoms_coh_o, atoms_coh_h)
    mdut.write_data(difficulty, exp_name, simulation_name, atoms_xyz)
    mdut.write_in(difficulty, exp_name, simulation_name, atoms_xyz)
    mdut.run_md(difficulty, exp_name, simulation_name)
    strain_array_go, stress_array_go, energy_go = utils.read_curve(f'{simulation_path}/{simulation_name}'
                                                                   f'/curve_{simulation_name}.txt')
    os.remove(f'{simulation_path}/{simulation_name}/lmp_serial')
    return energy_go, simulation_name



def run_graphene():
    atoms_fixed, atoms_move, atoms_relax = mdut.basal_plane()
    atoms_xyz = mdut.assemble_graphene(atoms_fixed, atoms_move, atoms_relax)
    mdut.write_data('_', atoms_xyz, False)
    mdut.write_in('_', atoms_xyz, False)
    mdut.run_md('_')
    strain_array_g, stress_array_g, energy_g = utils.read_curve('curve__.txt')
    os.remove('lmp_serial')
    os.chdir('..')
    return energy_g
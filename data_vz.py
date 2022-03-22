import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import utils
import copy


def load_results(difficulty: str):
    difficulty_name = f"DATA/{difficulty}"
    exp_names = sorted([exp_name for exp_name in os.listdir(difficulty_name)])
    # Define a dictionary to store all results
    all_results = {}
    # Load results in dataframes
    for exp_name in exp_names:
        result_path = f"{difficulty_name}/{exp_name}/{exp_name}.csv"
        try:
            result = pd.read_csv(result_path)
            all_results[exp_name] = result
            print(f'{exp_name} loaded! **** length: {len(result)}')
        except:
            if exp_name != "FIGURES":
                print(f'{exp_name} needs attention')
            continue
    return all_results


def load_random(difficulty: str, density_level: str):
    random_path = f"./DATA/{difficulty}/random_{density_level}/random_{density_level}.csv"
    result = pd.read_csv(random_path)["Reward"]
    return result.mean(), result.std()


def plot_results(difficulty: str, all_results: dict, keys: list, length: int,
                 label_name: str,
                 fill_between_plots=True,
                 # moving average
                 moving_ave=True, window=10,
                 # save the figure
                 save=False, fig_name="Fig",
                 # general settings
                 figsize=(8, 6), dpi=120, linewidth=1.5, use_legend=True,
                 first_fig=True, last_fig=True
                 ):

    fig_path = f"DATA/{difficulty}/FIGURES"
    # Calculate moving average of results
    results4plot = copy.deepcopy(all_results)
    x4plot = np.arange(length)
    if moving_ave:
        for key in keys:
            results4plot[key]["Reward"] = utils.moving_average(results4plot[key]["Reward"][:length], window=window)

    if first_fig:
        plt.figure(figsize=figsize, dpi=dpi)

    if fill_between_plots:
        curve_mean, curve_low, curve_high = utils.curves_mean_std(results4plot, keys, length)

        plt.fill_between(x=x4plot, y1=curve_low, y2=curve_high, alpha=.35)
        plt.plot(x4plot, curve_mean, linewidth=linewidth, label=label_name)
    else:
        for key in keys:
            plt.plot(x4plot, results4plot[key]["Reward"][:length], label=key, linewidth=linewidth)

    if last_fig:
        plt.title(f"{difficulty}".title(), fontsize=20)
        plt.xlabel("Iteration", fontsize=20)
        plt.ylabel("Return", fontsize=20)
        if use_legend:
            plt.legend(fontsize=16, loc="lower right", edgecolor="black")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        if save:
            plt.savefig(fname=f"{fig_path}/{fig_name}.png")

        plt.show()


def plot_stress_strain(curve: str, first_fig: bool, last_fig: bool, label_name: str,
                       ref_data=None, save=False, fig_name="MD_stress-strain_curve",
                       moving_ave=True, window=6, figsize=(8, 6), dpi=120):
    strain, stress, _ = utils.read_curve(curve)

    if moving_ave:
        stress = utils.moving_average(pd.Series(stress.reshape(-1)), window=window)

    if first_fig:
        plt.figure(figsize=figsize, dpi=dpi)

    plt.plot(strain, stress, linewidth=1.5, label=label_name)

    if last_fig:
        # plt.title("Stress-Strain Curves", fontsize=14)
        plt.xlabel("Strain", fontsize=14)
        plt.ylabel("Stress (GPa)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if ref_data is not None:
            ref_curve = pd.read_csv(ref_data)
            plt.plot(ref_curve["strain"], ref_curve["stress"], label="RSC Adv., 2020, 10, 29610")
        plt.legend(fontsize=12, loc="lower right", edgecolor="black")
        fig_path = "FIGURES"
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        if save:
            plt.savefig(fname=f"{fig_path}/{fig_name}.png")
        plt.show()
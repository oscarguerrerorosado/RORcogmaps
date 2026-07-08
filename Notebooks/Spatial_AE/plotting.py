import numpy as np
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d


from training import predict
from analysis import compute_mean_std

def plot_ratemaps(r, path, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,20), dpi=600)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(r[i], cmap='hot', origin='lower')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(path + '/100ratemaps.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(path + '/100units.png', format='png')
    plt.show()
#####################################################################

def plot_single_ratemap_density(r, unit, all_num_fields, sizes_per_field, centroids_per_field, plot_path, figsize=(3,3), save=False):
    print('Number of place fields = ' + str(all_num_fields[unit]))
    print('Size of place fields = ' + str(sizes_per_field[unit]))
    print('YX position of place fields = ' + str(centroids_per_field[unit]))

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(r[unit], cmap='hot', origin='lower')

    # Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Activation', rotation=270, labelpad=12)
    
    if centroids_per_field[unit] != [[[0, 0]]]:
        for i in range(len(centroids_per_field[unit])):
            plt.scatter(centroids_per_field[unit][i][1], centroids_per_field[unit][i][0], color='green', marker='x', s=30)
    if save:
        fig.savefig(plot_path + '/Example_place_field.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_path + '/Example_place_field.png', format='png')
    plt.show()
#####################################################################

def plot_polarmaps(p, plot_path, n_bins=20, n_cells_plot=30, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,16), dpi=600)
    
    for i in range(n_cells_plot):

        bottom = 0.4

        theta = np.linspace(0.0, 2*np.pi, n_bins, endpoint=False)
        radii = p[i]
        width = (2*np.pi) / (n_bins-1)

        ax = plt.subplot(5,6,i+1, polar=True)
        plt.title('Unit '+str(i+1))
        bars = ax.bar(theta, radii, width=width, bottom=bottom)
        ax.set_theta_zero_location("W")

        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.jet(r / 5.))
            bar.set_alpha(0.8)

    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/Polar_maps.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/Polar_maps.png', format='png')
    plt.show()
#####################################################################


def plot_place_field_hist(num_fields, plot_path, save=False):
    '''
    TO DO.
    '''
    place_field_counts = np.histogram(num_fields, bins=np.max(num_fields)+1, density=True)[0]
    plt.figure(figsize=(5,4))
    plt.bar(np.arange(np.max(num_fields)+1), place_field_counts, width=1, color='black', alpha=1, edgecolor='white')
    plt.xlabel('# place fields', fontsize=20)
    plt.ylabel('prob.', fontsize=20)
    plt.yticks(np.linspace(0,1,6), np.linspace(0,1,6).round(1), fontsize=18)
    plt.xticks(np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), fontsize=18)
    plt.ylim(0,1)
    sb.despine()
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/prob_place_field_histogram.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/prob_place_field_histogram.png', format='png')
    plt.show()
#####################################################################

def plot_history_loss(history_loss, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.plot(range(0, 100), history_loss[:100], color='blue', label='Tmaze v1')
    plt.plot(range(99, 200), history_loss[99:200], color='orange', label='Tmaze v2')
    plt.plot(range(199, 300), history_loss[199:], color='red', label='Tmaze v3')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    sb.despine()
    plt.legend()
    plt.title('History Loss', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/History_loss.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/History_loss.png', format='png')


#####################################################################################################

def plot_group_history_loss(mean, std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    segments = [
        (range(0, 100), mean[0:100], std[0:100], 'blue', 'Tmaze v1'),
        (range(99, 200),  mean[99:200], std[99:200], 'orange', 'Tmaze v2'),
        (range(199, 300), mean[199:], std[199:], 'red', 'Tmaze v3'),
    ]

    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    sb.despine()
    plt.legend()
    plt.title('Mean Reconstruction Loss', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Group_history_loss.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Group_history_loss.png', format='png')
        
#####################################################################################################

def plot_loss_history_by_experiment(all_history_loss, figsize=(6,4), save=False, path=''):
    fig, ax = plt.subplots(figsize=figsize)
    
    mean, std = compute_mean_std(all_history_loss)
    timesteps = np.arange(300)
    colors = cm.tab20.colors 
    
    # Individual series
    for i, series in enumerate(all_history_loss):
        ax.plot(timesteps, series, color=colors[i], linewidth=0.8, alpha=0.8)
    
    # Mean and std envelope
    ax.plot(timesteps, mean, color="green", linewidth=1.5, label="Mean", zorder=5)
    ax.fill_between(timesteps, mean - std, mean + std,
                    alpha=0.25, color="green", label="±1 std")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Mean Reconstruction Loss by Experiment", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    sb.despine()
    plt.tight_layout()
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/History_loss_by_exp.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/History_loss_by_exp.png', format='png')

#####################################################################################################

def plot_compare_loss_history(RO_mean, control_mean, RO_std, control_std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    plt.plot(control_mean, color='gray', label='Control', alpha=0.25)
    control_x = range(0, 300)
    plt.fill_between(control_x, control_mean - control_std, control_mean + control_std, alpha=0.10, color='gray')
    
    segments = [
        (range(0, 100), RO_mean[0:100], RO_std[0:100], 'blue', 'Tmaze v1'),
        (range(99, 200),  RO_mean[99:200], RO_std[99:200], 'orange', 'Tmaze v2'),
        (range(199, 300), RO_mean[199:], RO_std[199:], 'red', 'Tmaze v3'),
    ]

    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    sb.despine()
    plt.legend()
    plt.title('Mean Reconstruction Loss', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Compare_history_loss.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Compare_history_loss.png', format='png')

#####################################################################################################

def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

#####################################################################################################

def add_colorbar(fig, ax, im):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Adjust fraction & pad for size and spacing
    cbar.ax.tick_params(labelsize=8)  # Adjust tick label size

#####################################################################################################

def plot_active_units_count(num_active_units, nhidden, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.plot(range(1, 101), num_active_units[1:101], color='blue', label='Tmaze v1')
    plt.plot(range(100, 201), num_active_units[100:201], color='orange', label='Tmaze v2')
    plt.plot(range(200, 301), num_active_units[200:], color='red', label='Tmaze v3')
    plt.scatter(range(0, 1), num_active_units[:1], color='black', label='Pre', s=25)
    
    plt.xlabel('Epoch')
    plt.ylabel('Units (out of {})'.format(nhidden))
    sb.despine()
    plt.legend()
    plt.title('Active units count', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Active_units.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Active_units.png', format='png')

#####################################################################################################

def plot_group_active_units_count(mean, std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    segments = [
        (range(1, 102), mean[1:102], std[1:102], 'blue', 'Tmaze v1'),
        (range(101, 202),  mean[101:202], std[101:202], 'orange', 'Tmaze v2'),
        (range(201, 301), mean[201:], std[201:], 'red', 'Tmaze v3'),
    ]

    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.scatter(range(0, 1), mean[:1], color='black', label='Pre', s=25)

    plt.xlabel('Epoch')
    plt.ylabel('Units (out of 200)')
    sb.despine()
    plt.legend()
    plt.title('Active units count', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Active_units.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Active_units.png', format='png')

#####################################################################################################

def plot_active_units_count_by_experiment(all_num_active_units, figsize=(6, 4), save=False, path=''):
    fig, ax = plt.subplots(figsize=figsize)
    
    mean, std = compute_mean_std(all_num_active_units)
    timesteps = np.arange(301)
    colors = cm.tab20.colors 
    
    # Individual series
    for i, series in enumerate(all_num_active_units):
        ax.plot(timesteps, series, color=colors[i], linewidth=0.8, alpha=0.8)
    
    # Mean and std envelope
    ax.plot(timesteps, mean, color="green", linewidth=1.5, label="Mean", zorder=5)
    ax.fill_between(timesteps, mean - std, mean + std,
                    alpha=0.25, color="green", label="±1 std")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Units (out of 200)", fontsize=12)
    ax.set_title("Mean Number of Active Units by Experiment", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    sb.despine()
    plt.tight_layout()
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Active_units_by_exp.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Active_units_by_exp.png', format='png')

#####################################################################################################

def plot_compare_active_units_count(RO_mean, control_mean, RO_std, control_std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    #plt.scatter(range(0, 1), control_mean[:1], color='gray', label='Pre-Control', s=25)
    plt.plot(control_mean[1:], color='gray', label='Control', alpha=0.25)
    control_x = range(1, 301)
    plt.fill_between(control_x, control_mean[1:] - control_std[1:], control_mean[1:] + control_std[1:], alpha=0.10, color='gray')

    segments = [
        (range(1, 102), RO_mean[1:102], RO_std[1:102], 'blue', 'Tmaze v1'),
        (range(101, 202),  RO_mean[101:202], RO_std[101:202], 'orange', 'Tmaze v2'),
        (range(201, 301), RO_mean[201:], RO_std[201:], 'red', 'Tmaze v3'),
    ]

    #plt.scatter(range(0, 1), RO_mean[:1], color='black', label='Pre', s=25)
    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Units (out of 200)')
    sb.despine()
    plt.legend()
    plt.title('Active units count', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Compare_active_units.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Compare_active_units.png', format='png')

#####################################################################################################

def plot_unit_coverage(mean_unit_coverage, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.plot(range(1, 101), mean_unit_coverage[1:101], color='blue', label='Tmaze v1')
    plt.plot(range(100, 201), mean_unit_coverage[100:201], color='orange', label='Tmaze v2')
    plt.plot(range(200, 301), mean_unit_coverage[200:], color='red', label='Tmaze v3')
    plt.scatter(range(0, 1), mean_unit_coverage[:1], color='black', label='Pre', s=25)
    
    plt.xlabel('Epoch')
    plt.ylabel('Coverage (%)')
    sb.despine()
    plt.legend()
    plt.title('Mean Unit Coverage', fontsize=15)
    plt.show()
    
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Unit_coverage.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Unit_coverage.png', format='png')

#####################################################################################################

def plot_group_unit_coverage(mean, std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    segments = [
        (range(1, 102), mean[1:102], std[1:102], 'blue', 'Tmaze v1'),
        (range(101, 202),  mean[101:202], std[101:202], 'orange', 'Tmaze v2'),
        (range(201, 301), mean[201:], std[201:], 'red', 'Tmaze v3'),
    ]

    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.scatter(range(0, 1), mean[:1], color='black', label='Pre', s=25)

    plt.xlabel('Epoch')
    plt.ylabel('Coverage (%)')
    sb.despine()
    plt.legend()
    plt.title('Mean Unit Coverage', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Unit_coverage.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Unit_coverage.png', format='png')

#####################################################################################################

def plot_unit_coverage_by_experiment(all_mean_unit_coverage, figsize=(6, 4), save=False, path=''):
    fig, ax = plt.subplots(figsize=figsize)
    
    mean, std = compute_mean_std(all_mean_unit_coverage)
    timesteps = np.arange(301)
    colors = cm.tab20.colors 
    
    # Individual series
    for i, series in enumerate(all_mean_unit_coverage):
        ax.plot(timesteps, series, color=colors[i], linewidth=0.8, alpha=0.8)
    
    # Mean and std envelope
    ax.plot(timesteps, mean, color="green", linewidth=1.5, label="Mean", zorder=5)
    ax.fill_between(timesteps, mean - std, mean + std,
                    alpha=0.25, color="green", label="±1 std")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Coverage %", fontsize=12)
    ax.set_title("Mean Unit Coverage by Experiment", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    sb.despine()
    plt.tight_layout()
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Unit_coverage_by_exp.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Unit_coverage_by_exp.png', format='png')

#####################################################################################################

def plot_compare_unit_coverage(RO_mean, control_mean, RO_std, control_std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    #plt.scatter(range(0, 1), control_mean[:1], color='gray', label='Pre-Control', s=25)
    plt.plot(control_mean[1:], color='gray', label='Control', alpha=0.25)
    control_x = range(1, 301)
    plt.fill_between(control_x, control_mean[1:] - control_std[1:], control_mean[1:] + control_std[1:], alpha=0.10, color='gray')
        
    segments = [
        (range(1, 102), RO_mean[1:102], RO_std[1:102], 'blue', 'Tmaze v1'),
        (range(101, 202),  RO_mean[101:202], RO_std[101:202], 'orange', 'Tmaze v2'),
        (range(201, 301), RO_mean[201:], RO_std[201:], 'red', 'Tmaze v3'),
    ]

    #plt.scatter(range(0, 1), RO_mean[:1], color='black', label='Pre', s=25)
    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.xlabel('Epoch')
    plt.ylabel('Coverage (%)')
    sb.despine()
    plt.legend()
    plt.title('Mean Unit Coverage', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Compare_unit_coverage.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Compare_unit_coverage.png', format='png')

#####################################################################################################

def plot_activity_rep_loc(BC_rep_unit_sum_history, TL_rep_unit_sum_history, TC_rep_unit_sum_history, TR_rep_unit_sum_history, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.plot(BC_rep_unit_sum_history[1:], label='Bottom-Center', color='#1f77b4')
    plt.plot(TL_rep_unit_sum_history[1:], label='Top-Left', color='#ff7f0e')
    plt.plot(TC_rep_unit_sum_history[1:], label='Top-Center', color='#2ca02c')
    plt.plot(TR_rep_unit_sum_history[1:], label='Top-Right', color='#d62728')
    plt.scatter(range(0, 1), BC_rep_unit_sum_history[:1], color='#1f77b4', s=15)
    plt.scatter(range(0, 1), TL_rep_unit_sum_history[:1], color='#ff7f0e', s=15)
    plt.scatter(range(0, 1), TC_rep_unit_sum_history[:1], color='#2ca02c', s=15)
    plt.scatter(range(0, 1), TR_rep_unit_sum_history[:1], color='#d62728', s=15)
    
    plt.xlabel('Epoch')
    plt.ylabel('Sum activity')
    sb.despine()
    plt.title('Sum actv. at representative locations', fontsize=15)
    plt.legend()
    plt.show()
    
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Rep_locations_actv.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Rep_locations_actv.png', format='png')

#####################################################################################################

def plot_group_activity_rep_loc(all_BC_rep_unit_sum_history, all_TL_rep_unit_sum_history, all_TC_rep_unit_sum_history, all_TR_rep_unit_sum_history, figsize=(6,4), save=False, path=''):
    BC_mean, BC_std = compute_mean_std(all_BC_rep_unit_sum_history)
    TL_mean, TL_std = compute_mean_std(all_TL_rep_unit_sum_history)
    TC_mean, TC_std = compute_mean_std(all_TC_rep_unit_sum_history)
    TR_mean, TR_std = compute_mean_std(all_TR_rep_unit_sum_history)

    timesteps = np.arange(len(all_BC_rep_unit_sum_history[0]))
    fig = plt.figure(figsize=figsize)

    plt.plot(BC_mean[1:], label='Bottom-Center', color='#1f77b4')
    plt.fill_between(timesteps, BC_mean - BC_std, BC_mean + BC_std, alpha=0.25, color="#1f77b4")
    plt.plot(TL_mean[1:], label='Top-Left', color='#ff7f0e')
    plt.fill_between(timesteps, TL_mean - TL_std, TL_mean + TL_std, alpha=0.25, color="#ff7f0e")
    plt.plot(TC_mean[1:], label='Top-Center', color='#2ca02c')
    plt.fill_between(timesteps, TC_mean - TC_std, TC_mean + TC_std, alpha=0.25, color="#2ca02c")
    plt.plot(TR_mean[1:], label='Top-Right', color='#d62728')
    plt.fill_between(timesteps, TR_mean - TR_std, TR_mean + TR_std, alpha=0.25, color="#d62728", label="±1 std")
    plt.scatter(range(0, 1), BC_mean[:1], color='#1f77b4', s=15)
    plt.scatter(range(0, 1), TL_mean[:1], color='#ff7f0e', s=15)
    plt.scatter(range(0, 1), TC_mean[:1], color='#2ca02c', s=15)
    plt.scatter(range(0, 1), TR_mean[:1], color='#d62728', s=15)
    
    plt.xlabel('Epoch')
    plt.ylabel('Sum activity')
    sb.despine()
    plt.title('Sum actv. at representative locations', fontsize=15)
    plt.legend()
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Sum_actv_rep_locations.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Sum_actv_rep_locations.png', format='png')
        
#####################################################################################################

def plot_units_rep_loc(BC_rep_unit_count_history, TL_rep_unit_count_history, TC_rep_unit_count_history, TR_rep_unit_count_history, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.plot(BC_rep_unit_count_history[1:], label='Bottom-Center', color='#1f77b4')
    plt.plot(TL_rep_unit_count_history[1:], label='Top-Left', color='#ff7f0e')
    plt.plot(TC_rep_unit_count_history[1:], label='Top-Center', color='#2ca02c')
    plt.plot(TR_rep_unit_count_history[1:], label='Top-Right', color='#d62728')
    plt.scatter(range(0, 1), BC_rep_unit_count_history[:1], color='#1f77b4', s=15)
    plt.scatter(range(0, 1), TL_rep_unit_count_history[:1], color='#ff7f0e', s=15)
    plt.scatter(range(0, 1), TC_rep_unit_count_history[:1], color='#2ca02c', s=15)
    plt.scatter(range(0, 1), TR_rep_unit_count_history[:1], color='#d62728', s=15)
    
    plt.xlabel('Epoch')
    plt.ylabel('Freq.')
    sb.despine()
    plt.title('Active units at representative locations', fontsize=15)
    plt.legend()
    plt.show()

    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Rep_locations_units.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Rep_locations_units.png', format='png')

#####################################################################################################

def plot_group_units_rep_loc(all_BC_rep_unit_count_history, all_TL_rep_unit_count_history, all_TC_rep_unit_count_history, all_TR_rep_unit_count_history, figsize=(6,4), save=False, path=''):
    BC_mean, BC_std = compute_mean_std(all_BC_rep_unit_count_history)
    TL_mean, TL_std = compute_mean_std(all_TL_rep_unit_count_history)
    TC_mean, TC_std = compute_mean_std(all_TC_rep_unit_count_history)
    TR_mean, TR_std = compute_mean_std(all_TR_rep_unit_count_history)

    timesteps = np.arange(len(all_BC_rep_unit_count_history[0]))
    fig = plt.figure(figsize=figsize)

    plt.plot(BC_mean[1:], label='Bottom-Center', color='#1f77b4')
    plt.fill_between(timesteps, BC_mean - BC_std, BC_mean + BC_std, alpha=0.25, color="#1f77b4")
    plt.plot(TL_mean[1:], label='Top-Left', color='#ff7f0e')
    plt.fill_between(timesteps, TL_mean - TL_std, TL_mean + TL_std, alpha=0.25, color="#ff7f0e")
    plt.plot(TC_mean[1:], label='Top-Center', color='#2ca02c')
    plt.fill_between(timesteps, TC_mean - TC_std, TC_mean + TC_std, alpha=0.25, color="#2ca02c")
    plt.plot(TR_mean[1:], label='Top-Right', color='#d62728')
    plt.fill_between(timesteps, TR_mean - TR_std, TR_mean + TR_std, alpha=0.25, color="#d62728", label="±1 std")
    plt.scatter(range(0, 1), BC_mean[:1], color='#1f77b4', s=15)
    plt.scatter(range(0, 1), TL_mean[:1], color='#ff7f0e', s=15)
    plt.scatter(range(0, 1), TC_mean[:1], color='#2ca02c', s=15)
    plt.scatter(range(0, 1), TR_mean[:1], color='#d62728', s=15)
    
    
    plt.xlabel('Epoch')
    plt.ylabel('Freq.')
    sb.despine()
    plt.title('Active units at representative locations', fontsize=15)
    plt.legend()
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Sum_units_rep_locations.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Sum_units_rep_locations.png', format='png')
        
#####################################################################################################

def plot_cogmap_correlation(correlation_history, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.plot(range(1, 100), correlation_history[1:100], color='blue', label='Tmaze v1')
    plt.plot(range(99, 200), correlation_history[99:200], color='orange', label='Tmaze v2')
    plt.plot(range(199, 300), correlation_history[199:], color='red', label='Tmaze v3')
    plt.scatter(range(0, 1), correlation_history[:1], color='black', label='Pre', s=25)
    
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    sb.despine()
    plt.legend(loc=4)
    plt.title('Cog. Maps similarity (t vs. t–1)', fontsize=15)
    plt.show()
    
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Map_similarity.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Map_similarity.png', format='png')

#####################################################################################################

def plot_group_cogmap_correlation(mean, std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)

    segments = [
        (range(0, 100), mean[0:100], std[0:100], 'blue', 'Tmaze v1'),
        (range(99, 200),  mean[99:200], std[99:200], 'orange', 'Tmaze v2'),
        (range(199, 300), mean[199:], std[199:], 'red', 'Tmaze v3'),
    ]

    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    plt.scatter(range(0, 1), mean[:1], color='black', label='Pre', s=25)

    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    sb.despine()
    plt.legend(loc=4)
    plt.title('Cog. Maps Similarity (t vs. t–1)', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Map_similarity.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Map_similarity.png', format='png')

#####################################################################################################

def plot_cogmap_correlation_by_experiment(all_correlation_history, figsize=(6, 4), save=False, path=''):
    fig, ax = plt.subplots(figsize=figsize)
    
    mean, std = compute_mean_std(all_correlation_history)
    timesteps = np.arange(300)
    colors = cm.tab20.colors 
    
    # Individual series
    for i, series in enumerate(all_correlation_history):
        ax.plot(timesteps, series, color=colors[i], linewidth=0.8, alpha=0.8)
    
    # Mean and std envelope
    ax.plot(timesteps, mean, color="green", linewidth=1.5, label="Mean", zorder=5)
    ax.fill_between(timesteps, mean - std, mean + std,
                    alpha=0.25, color="green", label="±1 std")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title('Cog. Maps Similarity by Experiments(t vs. t–1)', fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    sb.despine()
    plt.tight_layout()
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Map_similarity_by_experiment.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Map_similarity_by_experiment.png', format='png')

#####################################################################################################

def plot_compare_cogmap_correlation(RO_mean, control_mean, RO_std, control_std, figsize=(6,4), save=False, path=''):
    fig = plt.figure(figsize=figsize)
    
    segments = [
        (range(0, 100), RO_mean[0:100], RO_std[0:100], 'blue', 'Tmaze v1'),
        (range(99, 200),  RO_mean[99:200], RO_std[99:200], 'orange', 'Tmaze v2'),
        (range(199, 300), RO_mean[199:], RO_std[199:], 'red', 'Tmaze v3'),
    ]

    #plt.scatter(range(0, 1), RO_mean[:1], color='black', label='Pre', s=25)
    for x, m, s, color, label in segments:
        x = list(x)
        plt.plot(x, m, color=color, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.25, color=color)

    #plt.scatter(range(0, 1), control_mean[:1], color='gray', label='Pre-Control', s=25)
    plt.plot(control_mean[1:], color='gray', label='Control', alpha=0.5)
    control_x = range(1, 300)
    plt.fill_between(control_x, control_mean[1:] - control_std[1:], control_mean[1:] + control_std[1:], alpha=0.10, color='gray')

    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    sb.despine()
    plt.legend(loc=4)
    plt.title('Cog. Maps Similarity (t vs. t–1)', fontsize=15)
    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/Compare_map_similarity.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/Compare_map_similarity.png', format='png')

#####################################################################################################

def plot_img_reconstruction(m, input_img, mt0, mt10, mt100, tmazev=0, figsize=(14,3), m0_bool=False, save=False, path=''):
    fig = plt.figure(figsize=figsize)
    plt.subplot(141)
    plt.imshow(input_img)
    plt.suptitle('Image reconstruction', fontsize=15)
    plt.title('Input', fontsize=10)
    plt.ylabel('Tmaze v{}'.format(tmazev))
    
    plt.subplot(142)
    plt.imshow(predict(input_img, mt0))
    if m0_bool == True:
        plt.title('Epoch 0', fontsize=10)
    else:
        plt.title('Epoch 1', fontsize=10)

    plt.subplot(143)
    plt.imshow(predict(input_img, mt10))
    plt.title('Epoch 10', fontsize=10)

    plt.subplot(144)
    plt.imshow(predict(input_img, mt100))
    plt.title('Epoch 100', fontsize=10)

    plt.show()

    if save == True:
        filename = 'Tmaze{}_img_recon'.format(m)
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/' + filename + '.png', format='png')

#####################################################################################################

def plot_example_ratemaps_progress(example_units, example_ratemaps, figsize=(15, 8), save=False, path=''):
    
    fig, axes = plt.subplots(5, 9, figsize=figsize)
    colors = ['blue'] * 3 + ['orange'] * 3 + ['red'] * 3
    ts = ['0', '10', '100', '1', '10', '100', '1', '10', '100']

    for i, ax in enumerate(axes.flatten()):
        # Determine the column index (0–8)
        col = i % 9
        color = colors[col]
    
        # Draw dashed T-maze using the color for this column
        ax.axhline(y=35, xmin=0, xmax=0.3, color=color, linestyle='--', linewidth=2)
        ax.axhline(y=35, xmin=0.7, xmax=1, color=color, linestyle='--', linewidth=2)
        ax.axhline(y=49, xmin=0, xmax=1, color=color, linestyle='--', linewidth=2)
        ax.axhline(y=0, xmin=0.3, xmax=0.7, color=color, linestyle='--', linewidth=2)
        ax.axvline(x=35, ymin=0, ymax=0.7, color=color, linestyle='--', linewidth=2)
        ax.axvline(x=15, ymin=0, ymax=0.7, color=color, linestyle='--', linewidth=2)
        ax.axvline(x=0, ymin=0.7, ymax=1, color=color, linestyle='--', linewidth=2)
        ax.axvline(x=49, ymin=0.7, ymax=1, color=color, linestyle='--', linewidth=2)
    
    for mt in range(len(example_ratemaps)):
        for unit in range(len(example_ratemaps[mt])):
            im = axes[unit, mt].imshow(example_ratemaps[mt][unit], cmap='hot', origin='lower')
            hide_ticks(axes[unit, mt])
            if mt == 0: axes[unit, mt].set_ylabel("Unit " + str(example_units[unit]), fontsize=15, labelpad=10)
            if unit == 0: axes[unit, mt].set_title('t' + ts[mt], fontsize=15)

    plt.show()
    if save == True:
        if path == '':
            print('Not saving - Saving path empty')
        else:
            fig.savefig(path + '/example_units.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(path + '/example_units.png', format='png')

#####################################################################################################

def plot_occupancy_map(XYposition, plot_folder, save=False):
    x_array = XYposition[:, 0]
    y_array = XYposition[:, 1]
    x_bins = 40
    y_bins = 40
    figsize = (7, 6)

    xedges = np.linspace(np.min(x_array), np.max(x_array), x_bins + 1)
    yedges = np.linspace(np.min(y_array), np.max(y_array), y_bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("auto")

    hist, xedges, yedges, im = ax.hist2d(x_array, y_array, bins=[xedges, yedges])

    cbar = fig.colorbar(im)
    cbar.set_label('Normal timesteps', rotation=270, fontsize=15, labelpad=25)

    ax.set_title('DoubleTmaze occupancy map', fontsize=15)
    ax.set_xticks(np.arange(np.min(x_array), np.max(x_array), step=.5))
    ax.set_yticks(np.arange(np.min(y_array), np.max(y_array), step=.25))

    plt.show()
    if save:
        fig.savefig(plot_folder + '/Occupancy_map.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_folder + '/Occupancy_map.png', format='png')

    return hist

#####################################################################################################

def plot_agent_trajectory(XYposition, plot_folder, save=False):

    obstacle1x =[-1.22, -1.22, -.45, -.45, -1.1, -1.1, -.45, -.45, -1.22, -1.22, -.07, -.07, .07, .07, 1.22, 1.22, .45,
                .45, 1.1, 1.1, .45, .45, 1.22, 1.22, .07, .07, -.07, -.07, -1.22]
    obstacle1y =[-1.2, -.8, -.8, -.7, -.7, .7, .7, .8, .8, 1.2, 1.2, .75, .75, 1.2, 1.2, .8, .8, .7, .7, -.7, -.7, 
                -.8, -.8, -1.2, -1.2, -.75, -.75, -1.2, -1.2]
    obstacle2x = [-.4, -.4, .4, .4, -.4]
    obstacle2y = [-.35, -.2, -.2, -.35, -.35]
    obstacle3x = [-.4, -.4, .4, .4, -.4]
    obstacle3y = [.35, .2, .2, .35, .35]
    obstaclesx = [obstacle1x, obstacle2x,obstacle3x]
    obstaclesy = [obstacle1y, obstacle2y,obstacle3y]

    figsize=(7, 6) 
    fig = plt.figure(figsize=figsize)

    for i in range(len(obstaclesx)):
        plt.plot(obstaclesx[i], obstaclesy[i], linewidth=3, color='grey')

    x_array = XYposition[:, 0]
    y_array = XYposition[:, 1]

    plt.plot(x_array, y_array, linewidth=0.2)
    plt.yticks(np.arange(-2, 2, step=.2), fontsize=10)
    plt.xticks(np.arange(-2, 2, step=.2), fontsize=10)
    plt.title("Agent's trajectory", fontsize=30)
    plt.plot(x_array[0],y_array[0],'ro', markersize=10)
    plt.show()
    
    if save:
        fig.savefig(plot_folder + '/Trajectory.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_folder + '/Trajectory.png', format='png')

#####################################################################################################

def plot_samples_per_bin_hist(hist, descending=True):
    counts = hist.flatten()

    # keep only visited bins (drop the zeros outside the maze arms)
    counts = counts[counts > 0]

    order = np.argsort(counts)
    if descending:
        order = order[::-1]
    counts = counts[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(counts)), counts, color='steelblue',
           edgecolor='white', linewidth=0.3)

    ax.set_title('Occupancy per spatial bin', fontsize=14)
    ax.set_xlabel('Spatial bin (ordered by occupancy)')
    ax.set_ylabel('Num. samples')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

#####################################################################################################

def value_heatmap(XYposition, values, title="", plot_path="", save=False):
    x_array = XYposition[:, 0]
    y_array = XYposition[:, 1]
    x_bins = 40
    y_bins = 40
    figsize = (7, 6)

    xedges = np.linspace(np.min(x_array), np.max(x_array), x_bins + 1)
    yedges = np.linspace(np.min(y_array), np.max(y_array), y_bins + 1)

    # Average value per bin instead of counting
    stat, xedges, yedges, _ = binned_statistic_2d(
        x_array, y_array, values, statistic='mean', bins=[xedges, yedges]
    )

    occupancy, _, _, _ = binned_statistic_2d(x_array, y_array, None, statistic='count', bins=[xedges, yedges])
    stat[occupancy < 5] = np.nan

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('auto')

    vmax = np.nanmax(np.abs(stat))  # symmetric colorscale
    im = ax.imshow(
        stat.T, origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='auto',
        cmap='RdBu_r',  # blue=negative, red=positive
        vmin=-vmax, vmax=vmax
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Relative value (with respect the mean)', rotation=270, fontsize=15, labelpad=25)
    ax.set_title(title, fontsize=15)
    plt.show()

    filename = title.replace(" ", "").replace(".", "")

    if save:
        if plot_path == "":
            print("Plot path is not defined")
        else:
            fig.savefig(plot_path + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
            fig.savefig(plot_path + '/' + filename + '.png', format='png')
    
    return stat
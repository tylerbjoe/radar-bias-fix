import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
import json
import re
import copy
from radarRespTrial import Radar_Resp
from collections import deque
import pandas as pd
"""
Extract respiration signals for a Lafayette data collection and plot them against RadarResp.
"""

def mean_filt_sig(sig, window_size):

    # Centered, not trailing

    if window_size % 2 == 0:  # EVEN  -  center is current and previous
        pre_win = int(window_size / 2)
        post_win = int(pre_win - 1)
    else:  # ODD  -  center is current
        pre_win = int(window_size // 2)
        post_win = int(window_size // 2)

    # Prepare the first window
    d_q = deque()
    sum_num = 0
    len_temp = 0
    for each_val in iter(sig[:post_win]):
        d_q.append(each_val)
        sum_num += each_val
        len_temp += 1
    result = []

    # Pre windows
    seq_pre = sig[int(post_win):int(pre_win + post_win)]
    for item in iter(seq_pre):
        d_q.append(item)
        sum_num += item
        len_temp += 1
        result.append(sum_num / len_temp)

    # Middle windows
    seq_mid = copy.deepcopy(sig[int(pre_win + post_win):])
    for item in iter(seq_mid):
        d_q.append(item)
        sum_num += item
        result.append(sum_num / window_size)
        sum_num -= d_q.popleft()

    # Post windows
    len_temp = len(d_q)
    while len(result) < sig.shape[0]:
        result.append(sum_num / len_temp)
        sum_num -= d_q.popleft()
        len_temp -= 1

    sig_filtered = np.array(result)
        
    return sig_filtered

def load_LX6(file_in):
    
    lx6_dict = {}
    cols_to_extract = ['Seconds(X)', 'P2 (Raw)', 'P1 (Raw)', 'EA (Raw)', 'EA (Automatic)', 'Annotations']
    col_inds = {}

    with open(file_in, "r") as f:
        fs_line = [l.strip() for l in f.readline().split(",")]
        lx6_dict['rate'] = ''
        for x in fs_line[0]:
            if x.isdigit():
                lx6_dict['rate'] += x
        lx6_dict['rate'] = float(lx6_dict['rate'])
        columns = [l.strip() for l in f.readline().split(",")]
        for each_col_ind, each_col in enumerate(columns):
            if each_col in cols_to_extract:
                col_inds[each_col] = each_col_ind
        txt_data = np.genfromtxt(file_in, skip_header=2, delimiter=",")
        for each_col in col_inds:
            if (each_col != 'Annotations'):
                if (col_inds[each_col] is not None):
                    lx6_dict[each_col] = txt_data[:, col_inds[each_col]]
                else:
                    lx6_dict[each_col] = np.array([np.nan] * txt_data.shape[0])
        lx6_dict['Annotations'] = np.genfromtxt(file_in, usecols=[col_inds['Annotations']], skip_header=2, delimiter=",", dtype=str)

    # Adjust for hardware sync
    pulse_loc = np.where(lx6_dict['Annotations'] == 'Sync')[0]
    if pulse_loc.size > 0:
        start_ind = pulse_loc[0]
    else:
        start_ind = 0
    pulse_time = lx6_dict['Seconds(X)'][start_ind]
    for key in lx6_dict:
        if key != 'rate':
            lx6_dict[key] = lx6_dict[key][start_ind:]
    lx6_dict['Seconds(X)'] -= pulse_time

    # Find annotation times
    annotation_dict = {}
    for i, annot in enumerate(lx6_dict['Annotations']):
        if annot:
            if annot not in annotation_dict:
                annotation_dict[annot] = []
            annotation_dict[annot].append(lx6_dict['Seconds(X)'][i])
    for annot in annotation_dict:
        annotation_dict[annot] = np.array(annotation_dict[annot])
    marker_types = list(annotation_dict.keys())
    question_bool = bool([bool(' Question' in x) for x in marker_types])
    if question_bool:
        qIDs = [re.findall(r'\(.*?\)', x)[0][1:-1] for x in marker_types if 'Start' in x]
        id_dict = {each_ID: [annotation_dict[f' QuestionStart ({each_ID})'][0], annotation_dict[f' QuestionEnd ({each_ID})'][0]] for each_ID in qIDs}
        remaining_markers = [x for x in marker_types if not ' Question' in x]
        for annot_remaining in remaining_markers:
            if '(' in annot_remaining:
                inner_id = re.findall(r'\(.*?\)', annot_remaining)[0][1:-1] if '(' in annot_remaining else annot_remaining
            elif annot_remaining == ' ':
                continue
            else:
                inner_id = annot_remaining
            id_dict[inner_id] = annotation_dict[annot_remaining]
        annotation_dict = copy.deepcopy(id_dict)

    plot_it = False
    # plot_it = True
    if plot_it:
        fig = plt.figure()
        fig_spec = gridspec.GridSpec(nrows=len(lx6_dict)-3, ncols=1, hspace=0.30)
        axs_dict = {}
        start_axs = 0
        for key in lx6_dict:
            if key not in ['rate', 'Seconds(X)', 'Annotations']:
                axs_dict[start_axs] = fig.add_subplot(fig_spec[start_axs])
                norm_sig = (lx6_dict[key]-lx6_dict[key].min())/(lx6_dict[key].max()-lx6_dict[key].min())
                axs_dict[start_axs].plot(lx6_dict['Seconds(X)'], norm_sig, label=key)
                axs_dict[start_axs].legend()
                for annot in annotation_dict:
                    if isinstance(annotation_dict[annot], list):
                        axs_dict[start_axs].axvspan(xmin=annotation_dict[annot][0], xmax=annotation_dict[annot][1], ymin=-1, ymax=+1, color='black', alpha=0.10)
                        axs_dict[start_axs].text(annotation_dict[annot][0], -1, annot, fontsize=8)
                    else:
                        for pt in annotation_dict[annot]:
                            axs_dict[start_axs].axvline(x=pt, ymin=-1, ymax=+1, color='black', linewidth=1)
                            axs_dict[start_axs].text(pt, -1, annot, fontsize=8)
                start_axs += 1
        plt.show(block=True)
            
    return lx6_dict, annotation_dict

def sort_folders(dir_in):
    subfolders = os.listdir(dir_in)
    file_list = []
    for each_subfolder in subfolders:
        sub_path = os.path.join(dir_in, each_subfolder)
        subcontents = os.listdir(sub_path)
        if 'meta.json' in subcontents:
            this_csv = [os.path.join(sub_path, x) for x in subcontents if '.csv' in x]
            file_list.extend(this_csv)
        else:
            for each_subsubfolder in subcontents:
                subsub_path = os.path.join(sub_path, each_subsubfolder)
                if os.path.isdir(subsub_path):
                    more_subcontents = os.listdir(subsub_path)
                    this_csv = [os.path.join(subsub_path, x) for x in more_subcontents if '.csv' in x]
                    file_list.extend(this_csv)
    return file_list

def plot_sigs(folder):

    lx6_file = os.path.join(folder, "lx6.csv")
    radar_chest_loc = os.path.join(folder, "lx6.npy")
    radar_abdo_loc = os.path.join(folder, "lx6.npy")

    file_name = os.path.basename(folder)
    new_path = os.path.join(folder, f"{file_name}.png")
        
    # Load LX6
    ldata, ldata_annotations = load_LX6(file_in=lx6_file)
    resp_chest = ldata['P1 (Raw)']
    resp_abdo = ldata['P2 (Raw)']
    resp_time = np.linspace(0, resp_chest.size/30, resp_chest.size)
    combined_resp = np.vstack((resp_chest, resp_abdo))
    resp_min = np.min(combined_resp)
    resp_max = np.max(combined_resp)
    resp_chest = (resp_chest - resp_min) / (resp_max - resp_min)
    resp_abdo = (resp_abdo - resp_min) / (resp_max - resp_min)
    resp_chest += 1
    combined_resp = np.vstack((resp_chest, resp_abdo))
        
    """
    Load radar signals here...
    """   
    radar_chest_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_1\Radar_2.csv"
    sigs = pd.read_csv(radar_chest_path, usecols=[0], header=None).squeeze().tolist()[1:]
    radar_chest = np.array([float(x) for x in sigs])
    radar_abdo_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_1\Radar_4.csv"
    sigs = pd.read_csv(radar_abdo_path, usecols=[0], header=None).squeeze().tolist()[1:]
    radar_abdo = np.array([float(x) for x in sigs])
    

    if len(radar_chest) > len(radar_abdo):
        radar_chest = radar_chest[:len(radar_abdo)]
    elif len(radar_abdo) > len(radar_chest):
        radar_abdo = radar_abdo[:len(radar_chest)]

    radar_time = np.linspace(0, radar_chest.size/30, radar_chest.size) 
    # Crop to just the questions
    # time_bounds = [60, 120]
    # inds = np.where(np.logical_and(resp_time > time_bounds[0], radar_time < time_bounds[1]))[0]
    # inds = np.where((resp_time > time_bounds[0]) & (resp_time < time_bounds[1]))[0]
    # radar_time = radar_time[inds]
    # radar_chest = radar_chest[inds]
    # radar_abdo = radar_abdo[inds]
    #
    radar_chest = radar_chest - mean_filt_sig(sig=radar_chest, window_size=(int(30*10)))
    radar_abdo = radar_abdo - mean_filt_sig(sig=radar_abdo, window_size=(int(30*10)))
    #
    radar_chest = (radar_chest - radar_chest.min()) / (radar_chest.max() - radar_chest.min())
    radar_abdo = (radar_abdo - radar_abdo.min()) / (radar_abdo.max() - radar_abdo.min())
    combined_radar_resp = np.vstack((radar_chest, radar_abdo))
    radar_resp_min = np.min(combined_radar_resp)
    radar_resp_max = np.max(combined_radar_resp)
    radar_chest = (radar_chest - radar_resp_min) / (radar_resp_max - radar_resp_min)
    radar_abdo = (radar_abdo - radar_resp_min) / (radar_resp_max - radar_resp_min)
    radar_chest += 1
    combined_radar_resp = np.vstack((radar_chest, radar_abdo))
    
    # Tick labels
    x_lims = [resp_time[0], resp_time[-1]]
    xticks = np.arange(int(x_lims[0]), int(x_lims[-1] + 1)).tolist()
    xticks = [x for x in xticks if x % 10 == 0]
    major_labels = []
    for each_tick in xticks:
        major_labels.append(each_tick)

    # Lafayette ytick labels
    resp_y_min = np.min(combined_resp)
    resp_y_max = np.max(combined_resp)
    resp_y_buff = 0.15 * (resp_y_max-resp_y_min)
    resp_y_lims = [resp_y_min-resp_y_buff, resp_y_max + resp_y_buff]
    resp_y_tick_diffs = (resp_y_lims[1]-resp_y_lims[0])/5
    resp_y_ticks = [np.round(resp_y_lims[0] + (i*resp_y_tick_diffs), 2) for i in np.arange(1, 5)]
    resp_major_y_labels = []
    for each_tick in resp_y_ticks:
        resp_major_y_labels.append('')

    # RadarResp ytick labels
    """
    Add RadarResp signals to the plot here instead of "moveresp"
    """
    # moveresp_signal = np.load(r"R:\Active Research\Projects\CANDOR\Data\10_01_24_ContactSensors\radar bias signals 1\signals - Copy\subject_1_radar.npy")
    
    # moveresp_y_min = np.min(moveresp_signal)
    # moveresp_y_max = np.max(moveresp_signal)
    # moveresp_y_buff = 0.15 * (moveresp_y_max-moveresp_y_min)
    # moveresp_y_lims = [moveresp_y_min-moveresp_y_buff, moveresp_y_max + moveresp_y_buff]
    # moveresp_y_tick_diffs = (moveresp_y_lims[1]-moveresp_y_lims[0])/5
    # moveresp_y_ticks = [np.round(moveresp_y_lims[0] + (i*moveresp_y_tick_diffs), 2) for i in np.arange(1, 5)]
    # moveresp_major_y_labels = []
    # for each_tick in moveresp_y_ticks:
    #     moveresp_major_y_labels.append('')

    # # Plotting
    # fig = plt.figure()
    # fig.set_size_inches(28, 16, True)
    # row_h = [1, 1]
    # spec = gridspec.GridSpec(ncols=1, nrows=int(len(row_h)), hspace=0.30, height_ratios=row_h)

    # # Plot lafayette
    # lafayette_axs  = fig.add_subplot(spec[0])
    # lafayette_axs.plot(resp_time, resp_chest, linewidth=4, label='P1 Thoracic', color='grey')
    # lafayette_axs.plot(resp_time, resp_abdo, linewidth=4, label='P2 Abdominal', color='blue')
    # lafayette_axs.grid(which='major', axis='y', color='grey', alpha=0.50, linewidth=1)  
    # lafayette_axs.set_ylim(resp_y_lims)
    # lafayette_axs.set_yticks(resp_y_ticks)
    # lafayette_axs.set_yticklabels(resp_major_y_labels)
    # lafayette_axs.set_title(f"Lafayette Polygraph System: Pneumograph", fontweight='bold', fontsize=30, pad=30)

    # # Plot MoveResp
    # moveresp_axs = fig.add_subplot(spec[1])
    # moveresp_axs.plot(radar_time, radar_chest, linewidth=4, label='RR Chest', color='black')
    # moveresp_axs.plot(radar_time, radar_abdo, linewidth=4, label='R Abdo', color='black')
    # moveresp_axs.grid(which='major', axis='y', color='grey', alpha=0.50, linewidth=1)  
    # # moveresp_axs.set_ylim(moveresp_y_lims)
    # # moveresp_axs.set_yticks(moveresp_y_ticks)
    # # moveresp_axs.set_yticklabels(moveresp_major_y_labels)
    # moveresp_axs.set_title(f"CANDOR: MoveResp", fontweight='bold', fontsize=30, pad=30)

    """
    Use the example contained in the laod_lx6 function to overlay the annotations onto the plots here
    """
    lx6_dict = ldata
    annotation_dict = ldata_annotations
    # Create figure
    fig = plt.figure()
    fig_spec = gridspec.GridSpec(nrows=len(lx6_dict) - 3, ncols=1, hspace=0.30)
    axs_dict = {}
    start_axs = 0
    for key in lx6_dict:
        if key not in ['rate', 'Seconds(X)', 'Annotations']:
            # Set up subplot
            axs_dict[start_axs] = fig.add_subplot(fig_spec[start_axs])
    
            # Normalize and plot data
            norm_sig = (lx6_dict[key] - lx6_dict[key].min()) / (lx6_dict[key].max() - lx6_dict[key].min())
            axs_dict[start_axs].plot(lx6_dict['Seconds(X)'], norm_sig, label=key)
            axs_dict[start_axs].set_ylabel('Displacement (unitless)')
            axs_dict[start_axs].legend()
    
            # Loop through annotations
            for annot in annotation_dict:
                if isinstance(annotation_dict[annot], list):
                    # Draw span for annotations with start and end points
                    start, end = annotation_dict[annot][0], annotation_dict[annot][1]
                    axs_dict[start_axs].axvspan(xmin=start, xmax=end, color='black', alpha=0.10)
    
                    # Place annotation text above the shaded span
                    axs_dict[start_axs].text(start, 1.025, annot, fontsize=9, ha='left', transform=axs_dict[start_axs].get_xaxis_transform())
                else:
                    # Draw lines for point annotations
                    for pt in annotation_dict[annot]:
                        axs_dict[start_axs].axvline(x=pt, color='black', linewidth=1)
    
                        # Place annotation text near each line
                        axs_dict[start_axs].text(pt+2, .92, annot, fontsize=9, ha='left', transform=axs_dict[start_axs].get_xaxis_transform())
    
            start_axs += 1
    
    # Set the xlabel on the last subplot
    axs_dict[0].set_title('Pneum Abdomen')
    axs_dict[1].set_title('Pneum Chest')
    axs_dict[start_axs - 1].set_xlabel('Time(s)')
    
    plt.show()
    
    # # Set xticks
    # for each_axs in fig.axes:
    #     each_axs.grid(which='major', axis='x', color='grey', alpha=0.50, linewidth=1)  
    #     each_axs.set_xlim(x_lims)
    #     each_axs.set_xticks(xticks)
    #     each_axs.set_xticklabels(major_labels)
    
    # # Save fig
    # fig.savefig(new_path, dpi=75)

if __name__ == '__main__':
    base_dir = r"R:\Active Research\Projects\CANDOR\Data\10_01_24_ContactSensors\s1_quest_phys"
    """
    Feed this function a folder with all the signals of interest and it'll generate a plot. 
    Modify as needed to be able to load the signals in their current form.
    """
    plot_sigs(base_dir)

import time
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from threading import Thread
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#%% NPI
class Radar_Resp_NPI():
    '''' Begin Class '''
    def __init__(self):
        self.fs = 30                            # frequency (Hz)
        self.win_size = 10 * self.fs            # window size (s)
        self.check_corr = .8 * self.fs          # rate to check if model still correlates to signal (s)
        self.window = np.zeros(self.win_size)   # empty window
        self.running = False                    # whether or not running
        self.n_peaks = []                       # empty array used to store all detected peaks
        self.sub_point = 0                      # signal point with model subtracted
        self.model_point = 0                    # interpolated peak point
        self.sample_n = 0                       # number of samples processed so far
        self.slope = 0                          # slope of model
        self.intercept = 0                      # intercept of model
        self.new_peak = False                   # whether or not a new peak is detected
        self.lastnpeak = [0,0]                  # x,y coordinates of last detected peak
        self.streak = 0                         # how many points in a row using same slope
        self.all_peaks = []
        
    def get_sub_point(self): # USE THIS TO GET A DETRENDED POINT BACK
        return self.sub_point
    
    def get_model_point(self):
        return self.model_point
    
    def get_npeaks(self):
        return self.n_peaks
    
    def get_all_peaks(self):
        return self.all_peaks
    
    def get_slope(self):
        return self.slope
    
    def set_sub_point(self, value):
        self.sub_point = value

    def set_model_point(self):
        self.model_point = self.slope * (self.streak) + self.intercept
            
    def set_intercept(self, value):
        self.intercept = value
        
    def add_data(self, value): # USE THIS TO ADD A DATA POINT
        self.sample_n += 1
        self.window = np.roll(self.window, -1)
        self.window[-1] = value
        #CHECK IF A NEW PEAK OCCURED
        if self.check_new_peak():
            self.streak = 0
        self.streak += 1
        if len(self.n_peaks) > 1:#take away
            self.set_model_point()
        self.set_sub_point(self.window[-1] - self.model_point)
        
    def check_new_peak(self):
        last_peaks = find_peaks(np.negative(self.window), prominence=250)[0]#1650 prominence when breathold #150 or 250 for other??? distance=2*30,, prominence=250
        if len(last_peaks) != 0:
            x = self.sample_n - (self.win_size - last_peaks[-1])
            y = self.window[last_peaks[-1]]
            self.all_peaks.append(x)
            if x != self.lastnpeak[0]:# and abs(x-self.lastnpeak[0]) >= 2.25 * self.fs:
                # if self.lastnpeak != [0,0]:
                slope = (y-self.lastnpeak[1])/(x-self.lastnpeak[0])
                self.lastnpeak=[x,y]
                if abs(slope) <= 29 or self.lastnpeak == [0,0]:#<+29 or 20
                    self.slope = slope
                    self.set_intercept(self.model_point)
                    # self.lastnpeak=[x,y]
                    self.n_peaks.append(x)
                    return True
                return False
            
    def clear_window(self):
        self.window = np.zeros(self.win_size)
        
    def reset(self):
        self.window = np.zeros(self.win_size)   # empty window
        self.running = False                    # whether or not running
        self.n_peaks = []                       # empty array used to store all detected peaks
        self.sub_point = 0                      # signal point with model subtracted
        self.model_point = 0                    # interpolated peak point
        self.sample_n = 0                       # number of samples processed so far
        self.slope = 0                          # slope of model
        self.intercept = 0                      # intercept of model
        self.new_peak = False                   # whether or not a new peak is detected
        self.lastnpeak = [0,0]                  # x,y coordinates of last detected peak
        self.streak = 0  
        self.all_peaks = []
        
    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        try:
            while self.running:
                print(f"{self.sub_point} Linear subtracted point")
                time.sleep(1/300000)
        except:
            self.running = False # Stop the thread if interrupted by keyboard (e.g., in Jupyter)

    def stop(self):
        self.running = False
    '''' End Class '''
    
#%% Load Data Imports
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import integrate

def get_summation(biny):
    total_bin = []
    total = 0
    for i in range(len(biny)):
       total += biny[i]
       total_bin.append(total)
    return total_bin

#%% CSVS 
title = "Bins 1-4 fast_breathing3 radar 1"
file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Pneumo.csv"
sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
truth = [int(x) for x in sigs]

file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Radar_2.csv"
sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
sig = [float(x) for x in sigs]
sig=np.array(sig[20:])
sig = sig[:len(truth)]

#%% SIGNAL PROCESSING
npi_sig = []
npi_mod = []
rr = Radar_Resp_NPI()
for val in sig:
    rr.add_data(val)
    npi_sig.append(rr.get_sub_point())
    npi_mod.append(rr.get_model_point())
all_peaks = rr.get_all_peaks()
used_peaks = rr.get_npeaks()

#%% Plotting
# time_axis = np.arange(len(sig)) / 30  # Time in seconds
# time_axis_npi = np.arange(len(npi_sig)) / 30  # Time in seconds
# time_axis_truth = np.arange(len(truth)) / 30  # Time in seconds
# # Create subplots
# fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
# fig.suptitle(title, fontsize=20, fontweight='bold')

# axes[0].set_title('Radar', fontsize=16)
# axes[0].set_ylabel('Displacement')
# # axes[0].plot(time_axis,sig, label='raw displacement', color='blue', linewidth=3)
# # axes[0].scatter([time_axis[idx] for idx in all_peaks],[sig[idx] for idx in all_peaks], label='negative peaks', color='green')
# # axes[0].scatter([time_axis[idx] for idx in used_peaks],[sig[idx] for idx in used_peaks], label='used negative peaks', color='orange')
# # axes[0].plot(time_axis_npi,npi_mod, label='npi model', color='orange', linewidth=1)
# axes[0].plot(time_axis_npi,npi_sig, label='displacement npi', color='red', linewidth=1)
# axes[0].grid()
# axes[0].legend(loc='upper right')
    
# axes[1].set_title('Pneum', fontsize=16)
# axes[1].set_ylabel('Displacement')
# axes[1].plot(time_axis_truth,truth, label='pneum', color='black', linewidth=1)
# axes[1].set_xlabel('Time (s)')
# axes[1].grid()
# axes[1].legend(loc='upper right')

#%% Sep plots
time_axis = np.arange(len(sig)) / 30  # Time in seconds

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
fig.suptitle(title, fontsize=20, fontweight='bold')

axes[0].set_title('Radar', fontsize=16)
axes[0].set_ylabel('Displacement')
axes[0].plot(time_axis,sig, label='raw displacement', color='blue', linewidth=3)
axes[0].scatter([time_axis[idx] for idx in all_peaks],[sig[idx] for idx in all_peaks], label='negative peaks', color='green')
axes[0].scatter([time_axis[idx] for idx in used_peaks],[sig[idx] for idx in used_peaks], label='used negative peaks', color='orange')
axes[0].plot(time_axis,npi_mod, label='npi model', color='orange', linewidth=1)
axes[0].grid()
axes[0].legend(loc='upper right')


axes[1].set_title('Detrended', fontsize=16)
axes[1].set_ylabel('Displacement')
axes[1].plot(time_axis,npi_sig, label='npi displacement', color='red', linewidth=1)
axes[1].grid()
axes[1].legend(loc='upper right')
    
axes[2].set_title('Pneum', fontsize=16)
axes[2].set_ylabel('Displacement')
axes[2].plot(time_axis,truth, label='pneum', color='black', linewidth=1)
axes[2].set_xlabel('Time (s)')
axes[2].grid()
axes[2].legend(loc='upper right')



#%% JSONS
# title = "Bins 1-4 fast_breathing3 radar 1"
# # "C:\Users\TJoe\Documents\Radar Offset Fix\close_range_testing_10_31\super_close_10_31\fast_breathing\fast_breathing3\Radar_1_metadata_1730403060.3604577.json"
# with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\close_range_testing_10_31\super_close_10_31\Top_of_Breath\ToB2\Radar_1_metadata_1730402495.8949175.json", 'r') as file:
#     json_data = json.load(file)
# bins = []
# df = pd.DataFrame(json_data)
# for i in range(6):
#     tmp_b, tmp_a = [], []
#     for j in range(1, len(df["frame_data"])):
#         tmp_b.append(df["frame_data"][j][i])
#     bins.append(tmp_b)
# biny = [sum(values) for values in zip(bins[1],bins[2],bins[3],bins[4])]
# integrated_bins = []

# for i in range(1,5):
#     integrated_bins.append(get_summation(bins[i]))
# integrated_bins.append(get_summation(biny))

# binss = [bins[1],bins[2],bins[3],bins[4],biny]
# #%% SIGNAL PROCESSING 
# rr = Radar_Resp_NPI()
# npi_bins = []
# peak_bins = []
# model_bins = []
# all_peak_bins = []
# for i in range(5):
#     holder = []
#     mholder = []
#     for val in integrated_bins[i]:
#         rr.add_data(val)
#         holder.append(rr.get_sub_point())
#         mholder.append(rr.get_model_point())
#     npi_bins.append(holder)
#     model_bins.append(mholder)
#     all_peak_bins.append(rr.get_all_peaks())
#     peak_bins.append(rr.get_npeaks())
#     holder=[]
#     mholder=[]
#     rr.reset()

# #%% Plot displacements

# time_axis = np.arange(len(bins[1])) / 30  # Time in seconds
# # Create subplots
# fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
# fig.suptitle(title, fontsize=20, fontweight='bold')
# for i in range(4):
#     axes[i].set_title(f'Bin {i + 1}', fontsize=16)
#     axes[i].set_ylabel('Displacement')
#     axes[i].plot(time_axis,integrated_bins[i], label='raw displacement', color='black', linewidth=3)
#     axes[i].scatter([time_axis[idx] for idx in all_peak_bins[i]],[integrated_bins[i][idx] for idx in all_peak_bins[i]], label='negative peaks', color='blue')
#     axes[i].scatter([time_axis[idx] for idx in peak_bins[i]],[integrated_bins[i][idx] for idx in peak_bins[i]], label='used negative peaks', color='orange')
#     axes[i].plot(time_axis,model_bins[i], label='npi model', color='darkorange', linewidth=1)
#     axes[i].plot(time_axis,npi_bins[i], label='displacement npi', color='red', linewidth=1)
#     axes[i].grid()
#     axes[i].legend(loc='upper right')
# axes[4].set_title('Bins 1-4', fontsize=16)
# axes[4].set_ylabel('Displacement')
# axes[4].plot(time_axis,integrated_bins[4], label='raw displacement', color='black', linewidth=3)
# axes[4].scatter([time_axis[idx] for idx in all_peak_bins[4]],[integrated_bins[4][idx] for idx in all_peak_bins[4]], label='negative peaks', color='blue')
# axes[4].scatter([time_axis[idx] for idx in peak_bins[4]],[integrated_bins[4][idx] for idx in peak_bins[4]], label='used negative peaks', color='orange')
# axes[4].plot(time_axis,model_bins[4], label='npi model', color='darkorange', linewidth=1)
# axes[4].plot(time_axis,npi_bins[4], label='displacement npi', color='red', linewidth=1)
# axes[4].grid()
# axes[4].legend(loc='upper right')


































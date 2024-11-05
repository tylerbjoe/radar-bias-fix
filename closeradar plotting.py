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
        self.fs = 30                    # frequency (Hz)
        
        self.win_size = 10 * self.fs     # window size (s)
        self.check_corr = .8 * self.fs   # rate to check if model still correlates to signal (s)
        self.window = np.zeros(self.win_size)               # empty window
        
        self.running = False            # whether or not running
        self.n_peaks = []  
        self.sub_point = 0              # signal point with model subtracted
        self.model_point = 0           # correlation metric (pearsonr)
        self.sample_n = 0               # number of samples processed
        self.slope = 0                  # slope of model
        self.intercept = 0              # intercept of model
        self.new_peak = False
        self.lastnpeak = [0,0]
        self.streak = 0
        
    def get_sub_point(self): # USE THIS TO GET A DETRENDED POINT BACK
        return self.sub_point
    
    def get_model_point(self):
        return self.model_point
    
    def get_npeaks(self):
        return self.n_peaks
    
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
        # if self.sample_n >= self.win_size:
        #CHECK IF A NEW PEAK OCCURED
        if self.sample_n % self.check_corr == 0:
            if self.check_new_peak():
                self.streak = 0
        self.streak += 1
        self.set_model_point()
        self.set_sub_point(self.window[-1] - self.model_point)
        
    def check_new_peak(self):
        last_peaks = find_peaks(np.negative(self.window), prominence=150)[0]#1650 prominence when breathold #250 for other??? distance=2*30,, prominence=250
        if len(last_peaks) != 0:
            x = self.sample_n - (self.win_size - last_peaks[-1])
            y = self.window[last_peaks[-1]]
            if x != self.lastnpeak[0] and abs(x-self.lastnpeak[0]) >= 2.25 * self.fs:
                # if self.lastnpeak != [0,0]:
                slope = (y-self.lastnpeak[1])/(x-self.lastnpeak[0])
                if abs(slope) <= 60 or self.lastnpeak == [0,0]:#<+29 or 20
                    self.slope = slope
                        
                    self.set_intercept(self.model_point)
                    self.lastnpeak=[x,y]
                    self.n_peaks.append(x)
                    return True
                return False
            
    def clear_window(self):
        self.window = np.zeros(self.win_size)
        
    def reset_sample_n(self):
        self.sample_n = 0;
        self.intercept = 0;

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
#%% LMS 
class Radar_Resp_LMS():
    '''' Begin Class '''
    def __init__(self):
        self.fs = 30                    # frequency (Hz)
        
        self.win_size = 12 * self.fs     # window size (s)
        self.check_corr = 0.3 * self.fs   # rate to check if model still correlates to signal (s)
        self.corr_threshold = 0.7       # correlation threshold to refresh model
        self.window = np.zeros(self.win_size)              # empty window
        
        self.running = False            # whether or not running
        self.lin_model = []             # empty linear model
        self.sub_point = 0              # signal point with model subtracted
        self.linear_point = 0
        self.correlation = 0            # correlation metric (pearsonr)
        self.sample_n = 0               # number of samples processed
        self.slope = 0                  # slope of model
        self.intercept = 0              # intercept of model
        self.lasty = 0
        self.streak = 0

    def get_sub_point(self): # USE THIS TO GET A DETRENDED POINT BACK
        return self.sub_point
    
    def set_sub_point(self):
        self.set_linear_point()
        self.sub_point =  self.window[-1] - self.linear_point

    def set_correlation(self):
        self.correlation,_ = pearsonr(self.window, (self.lin_model))
        
    def get_correlation(self):
        return self.correlation
    
    def add_data(self, value): # USE THIS TO ADD A DATA POINT
        self.sample_n += 1
        self.window = np.roll(self.window, -1)
        self.window[-1] = value
        if self.sample_n >= self.win_size:
            if self.sample_n != self.win_size:
                self.set_correlation()
            if (self.sample_n % self.check_corr == 0 and self.correlation<self.corr_threshold) or (self.win_size == self.sample_n):
                self.set_linear() # make new model if correlation too off and every x seconds
                self.streak = 0
            self.streak+=1
            self.set_sub_point()
        
    def set_linear(self):
        dur = len(self.window)
        x = np.linspace(0, dur/30, dur)
        self.intercept = self.linear_point
        self.slope, intercept, r, p, std_err = stats.linregress(x, self.window)
        if self.intercept == 0:
            self.intercept = intercept
        self.lin_model = self.slope * x + self.intercept
        
    def set_linear_point(self):
        if self.sample_n == self.win_size:
            self.linear_point = self.slope * (self.sample_n / self.fs) + self.intercept
            self.intercept = self.linear_point
        else:
            self.linear_point = self.slope * (self.streak / self.fs) + self.intercept
        
    def get_linear_point(self):
        return self.linear_point
    
    def reset_sample_n(self):
        self.sample_n = 0;
        self.intercept = 0;
    
    def clear_window(self):
        self.window = np.zeros(self.win_size)
    
    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        try:
            while self.running:
                print(f"{self.sub_point} Linear subtracted point")
                time.sleep(1/30000)
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
# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Pneumo.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
# truth = [int(x) for x in sigs][:15000]

# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Radar_2.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:][:15000]
# sig = [float(x) for x in sigs]
# sig=np.array(sig)


#%% JSONS
title = "Bins 1-4 fb3 r1"
with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\close_range_testing_10_31\super_close_10_31\fast_breathing\fast_breathing3\Radar_1_metadata_1730403060.3604577.json", 'r') as file:
    json_data = json.load(file)
bins = []
df = pd.DataFrame(json_data)
for i in range(6):
    tmp_b, tmp_a = [], []
    for j in range(1, len(df["frame_data"])):
        tmp_b.append(df["frame_data"][j][i])
    bins.append(tmp_b)
biny = [sum(values) for values in zip(bins[1],bins[2],bins[3],bins[4])]#,bins[5])]bins[0],
# biny = [sum(values) for values in zip(bins[0],bins[1],bins[2],bins[3],bins[4],bins[5])]
# sig = integrate.cumulative_trapezoid(biny)#get_summation(biny)#
integrated_bins = []

for i in range(1,5):
    integrated_bins.append(get_summation(bins[i]))
integrated_bins.append(get_summation(biny))


#%% Run Algs
npi_sig = []
lms_sig = []

rrNPI = Radar_Resp_NPI()
rrLMS = Radar_Resp_LMS()

for i in range(5):
    npi_bin_sig, lms_bin_sig = [], []
    
    # Process current bin
    for val in integrated_bins[i]:
        rrNPI.add_data(val)
        rrLMS.add_data(val)
        npi_bin_sig.append(rrNPI.get_sub_point())
        lms_bin_sig.append(rrLMS.get_sub_point())
    
    # Append processed signals for each bin to npi_sig and lms_sig
    npi_sig.append(npi_bin_sig)
    lms_sig.append(lms_bin_sig)
    
    # Clear window between each bin processing/reset sample number
    rrNPI.clear_window()
    rrLMS.clear_window()
    rrNPI.reset_sample_n()
    rrLMS.reset_sample_n()


#%% Plot
# time_axis = np.arange(len(bins[1])) / 30  # Time in seconds
time_axis = np.arange(len(integrated_bins[1])) / 30  # Time in seconds
# Create subplots
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True, sharey=True)  # 2 rows, 1 column
fig.suptitle(title, fontsize=20, fontweight='bold')
for i in range(4):
    axes[i].set_title(f'Bin {i+1}', fontsize=16)
    # axes[i].plot(time_axis,bins[i], label='raw velocity', color='black', linewidth=3)
    # axes[i].plot(time_axis,npi_sig[i], label="Detrended NPI", color='blue')
    # axes[i].plot(time_axis,lms_sig[i], label="Detrended LMS", color='darkorange')
    # axes[i].set_ylabel('Velocity')
    
    axes[i].plot(time_axis,integrated_bins[i], label='raw displacement', color='black', linewidth=3)
    axes[i].plot(time_axis,npi_sig[i], label="Detrended NPI", color='blue')
    axes[i].plot(time_axis,lms_sig[i], label="Detrended LMS", color='darkorange')
    axes[i].set_ylabel('Displacement (unitless)')
    
    axes[i].grid()
    axes[i].legend()
    


# 5th subplot: integrated bins 1-4
axes[4].set_title('Integrated 1-4', fontsize=16)
integrated_plot_sig = np.array(integrated_bins[4])/4
axes[4].plot(time_axis,integrated_plot_sig, label='raw displacement', color='black', linewidth=3)
axes[4].plot(time_axis,np.array(npi_sig[4])/4, label="Detrended NPI", color='blue')
axes[4].plot(time_axis,np.array(lms_sig[4])/4, label="Detrended LMS", color='darkorange')
axes[4].set_xlabel('Time (s)')
axes[4].set_ylabel('Displacement (unitless)')
axes[4].legend()
axes[4].grid()


plt.tight_layout()  # Adjust layout for better spacing
output_folder = r"C:\Users\TJoe\Documents\Radar Offset Fix\close range testing plots"
output_path = os.path.join(output_folder, f"{title}.png")
# fig.savefig(output_path, dpi=100)
plt.show()



#%% plot the 6 bins
# time_axis = np.arange(len(bins[1])) / 30  # Time in seconds
# # Create subplots
# fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
# fig.suptitle(title, fontsize=20, fontweight='bold')
# for i in range(6):
#     axes[i].set_title(f'Bin {i}', fontsize=16)
#     # axes[i].plot(time_axis,bins[i], label='raw velocity', color='black', linewidth=3)
#     # # axes[i].plot(time_axis,npi_sig[i], label="Detrended NPI", color='blue')
#     # axes[i].plot(time_axis,lms_sig[i], label="Detrended LMS", color='darkorange')
#     axes[i].set_ylabel('Velocity')
    
#     axes[i].plot(time_axis,bins[i], label='raw displacement', color='black', linewidth=3)
#     # axes[i].plot(time_axis,npi_sig[i], label="Detrended NPI", color='blue')
#     # axes[i].plot(time_axis,lms_sig[i], label="Detrended LMS", color='darkorange')
#     # axes[i].set_ylabel('Displacement (unitless)')
    
#     axes[i].grid()
#     axes[i].legend()

#%% different combinations of bins integrated
# time_axis = np.arange(len(integrated_bins[1])) / 30  # Time in seconds
# plt.figure()
# plt.title('Bin Combination Displacements', fontsize=16)
# plt.plot(integrated_bins[0], label='bins 0-5')
# plt.plot(integrated_bins[1], label='bins 1-4')
# plt.ylabel('Displacement (unitless)')
# plt.xlabel('Time (s)')
# plt.legend()
# plt.grid()










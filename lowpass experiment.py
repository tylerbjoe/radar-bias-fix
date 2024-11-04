import time
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from threading import Thread
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

#%%
import numpy as np
import copy
from collections import deque
from scipy import signal
 
##  DATA PROCESSING TOOLS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def butterworth_filter(sig: np.ndarray, fs: float, n_ord: int, low_cut: float=None, high_cut: float=None, pad_it: bool=True):
   
    """
    Method for appplying a digital butterworth filter to a signal.
 
    Parameters
    ----------
    sig : [1D np.array]
        The signal to be filtered.
    fs : [float]
        Sampling frequency.
    n_ord : [int]
        Filter order.
    low_cut : [float]
        Lower limit cutoff frequency (None for lowpass).
    high_cut : [float]
        Higher limit cutoff frequency (None for highpass).
    pad_it : [bool]
        Boolean indicator for whether to apply padding to the signal before filtering.
       
    Returns
    -------
    sig_filtered : [1D np.array]
        Filtered signal.
    """
 
    if (high_cut is None) and (low_cut is None):
        raise ValueError(f"Both high and low cut limits are None. Need to provide limits of some kind...")
    elif (high_cut is None):  # highpass
        [B, A] = signal.butter(N=n_ord, Wn=low_cut, fs=fs, btype='highpass', output='ba')
    elif (low_cut is None):  # highpass
        [B, A] = signal.butter(N=n_ord, Wn=high_cut, fs=fs, btype='lowpass', output='ba')
    else:  # bandpass
        [B, A] = signal.butter(N=n_ord, Wn=[low_cut, high_cut], fs=fs, btype='bandpass', output='ba')
    sig_filtered = sig.flatten()
    # if pad_it:
    #     ptype = 'odd'
    #     padding = 3 * (max([len(B), len(A)]) - 1)
    #     sig_filtered = signal.filtfilt(B, A, sig_filtered, padtype=ptype, axis=0, padlen=padding)
    # else:
    #     sig_filtered = signal.filtfilt(B, A, sig_filtered)
    zi = signal.lfilter_zi(B, A) * sig_filtered[0]
    sig_filtered = signal.lfilter(B, A, sig_filtered, zi=zi)[0]
   
    return sig_filtered
 
def mean_smoothing(sig: np.ndarray, fs: float, window_dur: float):
 
    """
    Method for smoothing the signal by returning the mean of a window centered at each point.
 
    Parameters
    ----------
    sig : [1D np.array]
        The signal to be filtered.
    fs : [float]
        Sampling frequency.
    window_dur : [float]
        Duration of the window over which to average the signal.
       
    Returns
    -------
    sig_filtered : [1D np.array]
        Filtered signal.
    """
 
    # Centered, not trailing
    window_size = int(window_dur*fs)
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

#%% LMS Alg
class Radar_Resp():
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

#%% NPI Alg
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
        
    def reset(self):
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
#%% CSVS
# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Pneumo.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
# truth = [int(x) for x in sigs][:15000]

# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Radar_2.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:][:15000]
# sig = [float(x) for x in sigs]
# sig=np.array(sig)


#%% JSONS
title = "Bins 1-4 fast_breathing3 radar 1"
with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\close_range_testing_10_31\super_close_10_31\fast_breathing\fast_breathing3\Radar_1_metadata_1730403060.3604577.json", 'r') as file:
    json_data = json.load(file)
bins = []
df = pd.DataFrame(json_data)
for i in range(6):
    tmp_b, tmp_a = [], []
    for j in range(1, len(df["frame_data"])):
        tmp_b.append(df["frame_data"][j][i])
    bins.append(tmp_b)
biny = [sum(values) for values in zip(bins[1],bins[2],bins[3],bins[4])]
integrated_bins = []

for i in range(1,5):
    integrated_bins.append(get_summation(bins[i]))
integrated_bins.append(get_summation(biny))

binss = [bins[1],bins[2],bins[3],bins[4],biny]
#%% SIGNAL PROCESSING 
filtered_bins = []
for b in binss:
    filtered_bins.append(butterworth_filter(np.array(b), fs=30, n_ord=2, low_cut=.15))

filtered_integrated_bins = []
for i in range(5):
    filtered_integrated_bins.append(get_summation(filtered_bins[i]))

window_mean_bins = []
for i in range(5):
    window_mean_bins.append(mean_smoothing(np.array(binss[i]), fs=30, window_dur=0.5))

integrated_window_mean_bins = []
for i in range(5):
    integrated_window_mean_bins.append(get_summation(window_mean_bins[i]))

rr = Radar_Resp_NPI()
lms_bins = []
for i in range(5):
    holder = []
    for val in filtered_integrated_bins[i]:
        rr.add_data(val)
        holder.append(rr.get_sub_point())
    lms_bins.append(holder)
    holder=[]
    rr.clear_window()
    rr.reset()

# integrated_lms_bins = []
# for i in range(5):
#     integrated_lms_bins.append(get_summation(lms_bins[i]))

#%% plot the 6 bins
time_axis = np.arange(len(bins[1])) / 30  # Time in seconds
# Create subplots
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
fig.suptitle(title, fontsize=20, fontweight='bold')
for i in range(4):
    axes[i].set_title(f'Bin {i + 1}', fontsize=16)
    axes[i].set_ylabel('Velocity')
    axes[i].plot(time_axis,binss[i], label='raw velocity', color='black', linewidth=3)
    axes[i].plot(time_axis,filtered_bins[i], label='10hz lp velocity', color='red', linewidth=1)
    # axes[i].plot(time_axis,window_mean_bins[i], label='window mean velocity', color='blue', linewidth=1)
    # axes[i].plot(time_axis,lms_bins[i], label='lms velocity', color='green', linewidth=1)
    axes[i].grid()
    axes[i].legend(loc='upper right')
axes[4].set_title('Bins 1-4', fontsize=16)
axes[4].set_ylabel('Velocity')
axes[4].plot(time_axis,binss[4], label='raw velocity', color='black', linewidth=3)
axes[4].plot(time_axis,filtered_bins[4], label='10hz lp velocity', color='red', linewidth=1)
# axes[4].plot(time_axis,window_mean_bins[4], label='window mean velocity', color='blue', linewidth=1)
# axes[4].plot(time_axis,lms_bins[4], label='lms velocity', color='green', linewidth=1)
axes[4].grid()
axes[4].legend(loc='upper right')



#%% Plot displacements

time_axis = np.arange(len(bins[1])) / 30  # Time in seconds
# Create subplots
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
fig.suptitle(title, fontsize=20, fontweight='bold')
for i in range(4):
    axes[i].set_title(f'Bin {i + 1}', fontsize=16)
    axes[i].set_ylabel('Displacement')
    axes[i].plot(time_axis,integrated_bins[i], label='raw displacement', color='black', linewidth=3)
    axes[i].plot(time_axis,filtered_integrated_bins[i], label='displacement .15hz hp velocity', color='red', linewidth=1)
    # axes[i].plot(time_axis,integrated_window_mean_bins[i], label='displacement win mean velocity', color='blue', linewidth=1)
    # axes[i].plot(time_axis,integrated_lms_bins[i], label='displacement lms velocity', color='green', linewidth=1)
    axes[i].plot(time_axis,lms_bins[i], label='lms filtered displacement', color='green', linewidth=1)
    axes[i].grid()
    axes[i].legend(loc='upper right')
axes[4].set_title('Bins 1-4', fontsize=16)
axes[4].set_ylabel('Displacement')
axes[4].plot(time_axis,integrated_bins[4], label='raw displacement', color='black', linewidth=3)
axes[4].plot(time_axis,filtered_integrated_bins[4], label='displacement .15hz hp velocity', color='red', linewidth=1)
# axes[4].plot(time_axis,integrated_window_mean_bins[4], label='displacement win mean velocity', color='blue', linewidth=1)
# axes[4].plot(time_axis,integrated_lms_bins[4], label='displacement lms velocity', color='green', linewidth=1)
axes[4].plot(time_axis,lms_bins[4], label='lms filtered displacement', color='green', linewidth=1)
axes[4].grid()
axes[4].legend(loc='upper right')







































# time_axis = np.arange(len(integrated_bins[1])) / 30  # Time in seconds
# # Create subplots
# fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True, sharey=True)  # 2 rows, 1 column
# fig.suptitle(title, fontsize=20, fontweight='bold')
# for i in range(4):
#     axes[i].set_title(f'Bin {i+1}', fontsize=16)

#     axes[i].plot(time_axis,integrated_bins[i], label='raw displacement', color='black', linewidth=3)
#     axes[i].set_ylabel('Displacement (unitless)')
    
#     axes[i].grid()
#     axes[i].legend()
    


# # 5th subplot: integrated bins 1-4
# axes[4].set_title('Integrated 1-4', fontsize=16)
# integrated_plot_sig = np.array(integrated_bins[4]) / 4
# axes[4].plot(time_axis,integrated_plot_sig, label='raw displacement', color='black', linewidth=3)
# axes[4].set_xlabel('Time (s)')
# axes[4].set_ylabel('Displacement (unitless)')
# axes[4].legend()
# axes[4].grid()


# plt.tight_layout()  # Adjust layout for better spacing
# output_folder = r"C:\Users\TJoe\Documents\Radar Offset Fix\close range testing plots"
# output_path = os.path.join(output_folder, f"{title}.png")
# plt.show()










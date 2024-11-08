import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import integrate
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from scipy import integrate
from scipy import stats
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import scipy.stats
from scipy.stats import pearsonr
# windowing function takes array and returns array of arrays containing windows
def get_windows(signal, window_len=10, overlap=2, fz=30):
    windows = []
    win_dur = int(window_len * fz)
    win_diff = int((window_len-overlap) * fz)
    #loop through sinal to get windows
    i = 0
    while i < len(signal) - win_dur + 1:
        win = signal[i : i + win_dur]
        windows.append(win)
        i += win_diff
    return windows

# linear model fit to window using least squares
def get_linear(signal):
    dur = len(signal)
    x = np.linspace(0, dur/30, dur)
    slope, intercept, r, p, std_err = stats.linregress(x, signal)
    return slope * x + intercept
   
def get_peakline(signal):
    last_peaks = find_peaks(-signal, prominence=550)[0]#1650 for breathold  distance=2.5*30,
    # x1 = last_peaks[-2]
    # y1 = signal[x1]
    # x2 = last_peaks[-1]
    # y2 = signal[x2]
    # signal = (y2-y1)/(x2-x1)
    if len(last_peaks) > 0:
        return last_peaks
    else:
        return 0
#%% LMS alg
class Radar_Resp_LMS():
    '''' Begin Class '''
    def __init__(self):
        self.fs = 30                    # frequency (Hz)
        
        self.win_size = 13 * self.fs     # window size (s)
        self.check_corr = 2.5 * self.fs   # rate to check if model still correlates to signal (s)
        # self.corr_threshold = 0.7       # correlation threshold to refresh model
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
    
    def set_win_size(self, value):
        self.win_size = value * self.fs
        self.window = np.zeros(self.win_size)
        
    def set_check_corr(self, value):
        self.check_corr = value * self.fs
        
    def set_corr_threshold(self, value):
        self.corr_threshold = value
    
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
            # if self.sample_n != self.win_size:
            #     self.set_correlation()
            if (self.sample_n % self.check_corr == 0) or (self.win_size == self.sample_n):# and self.correlation<self.corr_threshold
                self.set_linear() # make new model if correlation too off and every x seconds
                self.streak = 0
            self.streak+=1
            self.set_sub_point()
        
    def set_linear(self):
        
        # dur = len(self.window)
        x = np.linspace(0, self.win_size/30, self.win_size)
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
        
    def reset(self):

        self.fs = 30                    # frequency (Hz)
        
        self.win_size = 13 * self.fs     # window size (s)
        self.check_corr = 0.1 * self.fs   # rate to check if model still correlates to signal (s)
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
        self.ttime = 0
    
    
    # def start(self):
    #     self.running = True
    #     Thread(target=self._run, daemon=True).start()
    
    # def _run(self):
    #     try:
    #         while self.running:
    #             print(f"{self.sub_point} Linear subtracted point")
    #             # time.sleep(1/30000)
    #     except:
    #         self.running = False # Stop the thread if interrupted by keyboard (e.g., in Jupyter)

    # def stop(self):
    #     self.running = False
    '''' End Class '''


#%% NPI alg
from scipy.signal import find_peaks

class Radar_Resp_NPI():
    '''' Begin Class '''
    def __init__(self):
        self.fs = 30                            # frequency (Hz)
        
        self.win_size = 10 * self.fs            # window size (s)
        self.prominence = 100
        self.slope_limit = 20
        
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
        
    def set_win_size(self, value):
        self.win_size = value
        
    def set_prominence(self, value):
        self.prominence = value        
        
    def set_slope_limit(self, value):
        self.slope_limit = value  
        
    def add_data(self, value): # USE THIS TO ADD A DATA POINT
        self.sample_n += 1
        self.window = np.roll(self.window, -1)
        self.window[-1] = value
        #CHECK IF A NEW PEAK OCCURED
        if self.check_new_peak():
            self.streak = 0
        self.streak += 1
        if len(self.n_peaks) > 1:
            self.set_model_point()
        self.set_sub_point(self.window[-1] - self.model_point)
        
    def check_new_peak(self):
        last_peaks = find_peaks(np.negative(self.window), prominence=self.prominence)[0]#1650 prominence when breathold #150 or 250 for other??? distance=2*30,, prominence=250
        if len(last_peaks) != 0:
            x = self.sample_n - (self.win_size - last_peaks[-1])
            y = self.window[last_peaks[-1]]
            self.all_peaks.append(x)
            if x != self.lastnpeak[0]:# and abs(x-self.lastnpeak[0]) >= 2.25 * self.fs:
                # if self.lastnpeak != [0,0]:
                slope = (y-self.lastnpeak[1])/(x-self.lastnpeak[0])
                self.lastnpeak=[x,y]
                if abs(slope) <= 20 or self.lastnpeak == [0,0]:#<+29 or 20
                    self.slope = slope
                    self.set_intercept(self.model_point)
                    # self.lastnpeak=[x,y]
                    self.n_peaks.append(x)
                    return True
                return False
            
    def clear_window(self):
        self.window = np.zeros(self.win_size)
        
    def reset(self):
        self.win_size = 10 * self.fs            # window size (s)
        self.prominence = 450
        self.slope_limit = 20
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
        
    # def start(self):
    #     self.running = True
    #     Thread(target=self._run, daemon=True).start()
    
    # def _run(self):
    #     try:
    #         while self.running:
    #             print(f"{self.sub_point} Linear subtracted point")
    #             time.sleep(1/300000)
    #     except:
    #         self.running = False # Stop the thread if interrupted by keyboard (e.g., in Jupyter)

    # def stop(self):
    #     self.running = False
    '''' End Class '''
#%% JSONS
with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\distance_angle_testing\Andrew\close\ToB1\Radar_1_metadata_1730473210.9425592.json", 'r') as file:
    json_data = json.load(file)

bins = []

df = pd.DataFrame(json_data)
for i in range(6):
    tmp_b, tmp_a = [], []
    for j in range(1, len(df["frame_data"])):
        tmp_b.append(df["frame_data"][j][i])
    bins.append(tmp_b)

biny = [sum(values) for values in zip(bins[0],bins[1],bins[2],bins[3],bins[4],bins[5])]
sig = integrate.cumulative_trapezoid(biny)
#%% CSVS
# subjects = ['Subject_1','Subject_2','Subject_3','Subject_4_phys','Subject_4_quest','Subject_5','Subject_6','Subject_7_phys','Subject_7_quest','Subject_8_phys','Subject_8_quest','Subject_9_phys','Subject_9_quest']
# radars = ['Radar_2','Radar_4']



# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_5\Pneumo.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
# truth = [int(x) for x in sigs]

# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_5\Radar_4.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
# sig = [float(x) for x in sigs]
# sig = np.array(sig)
# sig = sig[:len(truth)]
# # script to run
# fs = 18
#%% loop!
# subjects = ['Subject_1','Subject_2','Subject_3','Subject_4_phys','Subject_4_quest','Subject_5','Subject_6','Subject_7_phys','Subject_7_quest','Subject_8_phys','Subject_8_quest','Subject_9_phys','Subject_9_quest']
# radars = ['Radar_2','Radar_4']
# param_dict = {1/30:0,.05:0}

# for subject in subjects:
#     for radar in radars:
#         file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_1\Pneumo.csv"
#         file_path = file_path.replace("Subject_1", subject)
#         sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
#         truth = [int(x) for x in sigs]
        
#         file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_2\Radar_2.csv"
#         file_path = file_path.replace("Subject_1", subject).replace("Radar_2", radar)
#         sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
#         sig = [float(x) for x in sigs]
#         sig = np.array(sig)
#         sig = sig[:len(truth)]
        
#         windows = get_windows(sig)
#         window=[]
#         time=[]
#         for win in windows:
#             for val in win:
#                 window.append(val)
#             lin = get_peakline(win)
#             times = (np.arange(start=len(time), stop=len(win))/30)
#             for t in times:
#                 time.append(t)
#             lin = 0
            
#         peaks = get_peakline(sig)
#         sig = np.array(sig)
        
#         negative_peak_values = sig[peaks]
        
#         interpolator = interp1d(peaks, negative_peak_values, kind='linear', fill_value="extrapolate")
#         interpolated_signal = interpolator(np.arange(len(sig)))
        
#         offline_npi = sig-interpolated_signal
        
#         lms_sigs=[]
#         lms_sig=[]
#         # corrs = []
#         rr_lms = Radar_Resp_LMS()
        
#         sig = sig[0:330*30]
#         offline_npi = offline_npi[:len(sig)]
#         truth = truth[:len(sig)]
#         for param in param_dict:
#             rr_lms.set_check_corr(param)
#             for val in sig:
#                 rr_lms.add_data(val)
#                 lms_sig.append(rr_lms.get_sub_point())
#             rr_lms.reset()
#             param_dict[param] += scipy.stats.pearsonr(lms_sig, offline_npi)[0]
#             lms_sigs.append(lms_sig)
#             lms_sig = []
#         print(f'{subject} {radar} done')
    
    
    
#%% WINDOWING
windows = get_windows(sig)
window=[]
time=[]
for win in windows:
    for val in win:
        window.append(val)
    lin = get_peakline(win)
    times = (np.arange(start=len(time), stop=len(win))/30)
    for t in times:
        time.append(t)
    lin = 0

#%% Offline PROCESSING
peaks = get_peakline(sig)
sig = np.array(sig)

negative_peak_values = sig[peaks]

interpolator = interp1d(peaks, negative_peak_values, kind='linear', fill_value="extrapolate")
interpolated_signal = interpolator(np.arange(len(sig)))

offline_npi = sig-interpolated_signal
#%% Alg Process

params = [50,100,150,200,250,300,350,400,450,500]
npi_sigs=[]
npi_sig=[]
corrs = []
rr_npi = Radar_Resp_NPI()

sig = sig[0:]#330*30
offline_npi = offline_npi[:len(sig)]
# truth = truth[:len(sig)]
for param in params:
    rr_npi.set_prominence(param)
    for val in sig:
        rr_npi.add_data(val)
        npi_sig.append(rr_npi.get_sub_point())
    rr_npi.reset()
    corrs.append(scipy.stats.pearsonr(npi_sig, offline_npi)[0])
    npi_sigs.append(npi_sig)
    npi_sig = []

params = [1/30,2.5,3.5]
lms_sigs=[]
lms_sig=[]
corrs = []
rr_lms = Radar_Resp_LMS()

sig = sig[0:330*30]
offline_npi = offline_npi[:len(sig)]
# truth = truth[:len(sig)]
for param in params:
    rr_lms.set_check_corr(param)
    for val in sig:
        rr_lms.add_data(val)
        lms_sig.append(rr_lms.get_sub_point())
    rr_lms.reset()
    corrs.append(scipy.stats.pearsonr(lms_sig, offline_npi)[0])
    lms_sigs.append(lms_sig)
    lms_sig = []


#%% Results
time_axis = np.arange(len(sig)) / 30
# Create subplots
fig, axes = plt.subplots(3, 1, sharex=True)  # 2 rows, 1 column

# First subplot: Signal and Linear Model
axes[0].plot(time_axis,sig, label=f'Signal r:{scipy.stats.pearsonr(sig, offline_npi)[0]:.5f}', color='blue')#time_axis, 
axes[0].set_ylabel('Displacement (unitless)')
axes[0].legend(loc='upper right')
axes[0].grid()
axes[0].set_title('Radar', fontsize=16)

for i in range(len(params)):
    axes[1].plot(time_axis, lms_sigs[i], label=f"var:{params[i]:.3f} r:{corrs[i]:.5f}")

axes[1].set_ylabel('Displacement (unitless)')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0))#legend(loc='upper right')
axes[1].grid()
axes[1].set_title('Real-Time Detrended', fontsize=16)

hcorr = corrs.index(max(corrs))
axes[2].plot(time_axis,lms_sigs[hcorr], label=f'rt lms var:{params[hcorr]:.3f}', color='darkorange')
axes[2].plot(time_axis,sig, label='raw radar', color='blue')
axes[2].plot(time_axis,offline_npi, label=' offline detrended', color='red')
axes[2].set_ylabel('Displacement (unitless)')
axes[2].legend()
axes[2].grid()
axes[2].set_title('Offline Detrended', fontsize=16)

# Second subplot: Correlation
# axes[3].plot(time_axis,truth, label='Pneum', color='black')
# axes[3].set_xlabel('Time')
# axes[3].set_ylabel('Displacement (unitless)')
# axes[3].legend()
# axes[3].grid()
# axes[3].set_title('Pneum', fontsize=16)

# plt.tight_layout()  # Adjust layout for better spacing
plt.show()




#%%
# time_axis = np.arange(len(sig)) / 30
# # Create subplots
# fig, axes = plt.subplots(4, 1, sharex=True)  # 2 rows, 1 column

# # First subplot: Signal and Linear Model
# axes[0].plot(time_axis,sig, label=f'Signal r:{scipy.stats.pearsonr(sig, offline_npi)[0]:.5f}', color='blue')#time_axis, 
# axes[0].set_ylabel('Displacement (unitless)')
# axes[0].legend(loc='upper right')
# axes[0].grid()
# axes[0].set_title('Radar', fontsize=16)

# for i in range(len(params)):
#     axes[1].plot(time_axis, npi_sigs[i], label=f"var:{params[i]} r:{corrs[i]:.5f}")

# axes[1].set_ylabel('Displacement (unitless)')
# axes[1].legend(loc='center left', bbox_to_anchor=(1, 0))#legend(loc='upper right')
# axes[1].grid()
# axes[1].set_title('Real-Time Detrended', fontsize=16)

# hcorr = corrs.index(max(corrs))
# axes[2].plot(time_axis,npi_sigs[hcorr], label=f'rt npi var:{params[hcorr]}', color='darkorange')
# axes[2].plot(time_axis,sig, label='raw radar', color='blue')
# axes[2].plot(time_axis,offline_npi, label=' offline detrended', color='red')
# axes[2].set_ylabel('Displacement (unitless)')
# axes[2].legend()
# axes[2].grid()
# axes[2].set_title('Offline Detrended', fontsize=16)

# # Second subplot: Correlation
# axes[3].plot(time_axis,truth, label='Pneum', color='black')
# axes[3].set_xlabel('Time')
# axes[3].set_ylabel('Displacement (unitless)')
# axes[3].legend()
# axes[3].grid()
# axes[3].set_title('Pneum', fontsize=16)

# plt.tight_layout()  # Adjust layout for better spacing
# plt.show()





























#%% PLOTTING SINGLE
# fs=18
# plt.figure()
# # plt.plot(time,window, label='signal')
# # plt.plot(time,lin, label="lms model")
# # plt.plot(time,subtracted, label="signal linear subtracted")
# plt.plot(time,sig, label='Signal',color='blue')
# plt.scatter(time[peaks],sig[peaks], color='orange')
# plt.plot(time,interpolated_signal, label='Interpolated Negative Peaks', linestyle='--', color='orange')#[peaks[0]:peaks[-1]]
# plt.plot(time,sig-interpolated_signal, label='detrended signal', color='red')

# #plt.title('8 Second window', fontsize=fs)
# plt.ylabel('displacement', fontsize=fs)
# plt.xlabel('time (s)', fontsize=fs)
# plt.legend(fontsize=fs)
# plt.grid()
# plt.show()
   
#%% SUBPLOTS W TRUTH
# time_axis = np.arange(len(sig)) / 30
# # Create subplots
# fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column

# # First subplot: Signal and Linear Model
# # axes[0].plot(time_axis,sig, label='Signal', color='blue')#time_axis, 
# # axes[0].plot(time_axis,interpolated_signal, label='Interpolated Negative Peaks', linestyle='--', color='orange')
# axes[0].plot(time_axis,sig-interpolated_signal, label='detrended signal', color='red')

# axes[0].set_ylabel('Displacement (unitless)')
# axes[0].legend()
# axes[0].grid()
# axes[0].set_title('Radar', fontsize=16)

# # Second subplot: Correlation
# # axes[1].plot(time_axis, corr, label='Correlation', color='green')
# axes[1].plot(time_axis,truth, label='Pneum', color='black')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Displacement (unitless)')
# axes[1].legend()
# axes[1].grid()
# axes[1].set_title('Pneum', fontsize=16)

# plt.tight_layout()  # Adjust layout for better spacing
# plt.show()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
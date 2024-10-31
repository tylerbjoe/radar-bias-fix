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
        last_peaks = find_peaks(np.negative(self.window), prominence=250)[0]#1650 prominence when breathold #250 for other??? distance=2*30,, prominence=250
        if len(last_peaks) != 0:
            x = self.sample_n - (self.win_size - last_peaks[-1])
            y = self.window[last_peaks[-1]]
            if x != self.lastnpeak[0] and abs(x-self.lastnpeak[0]) >= 2.25 * self.fs:
                # if self.lastnpeak != [0,0]:
                slope = (y-self.lastnpeak[1])/(x-self.lastnpeak[0])
                if abs(slope) <= 20 or self.lastnpeak == [0,0]:#<+29 or 20
                    self.slope = slope
                        
                    self.set_intercept(self.model_point)
                    self.lastnpeak=[x,y]
                    self.n_peaks.append(x)
                    return True
                return False

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
        self.window = []                # empty window
        
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
        if len(self.window) >= self.win_size:
            self.window.pop(0)
        self.window.append(value)
        if len(self.window) == self.win_size:
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
#%% Load Data
import json
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Pneumo.csv"
sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
truth = [int(x) for x in sigs][:15000]

file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_9_quest\Radar_2.csv"
sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:][:15000]
sig = [float(x) for x in sigs]
sig=np.array(sig)

#%% Run Algs
npi_sig = []
lms_sig = []

npi_mod = []
lms_mod = []

rrNPI = Radar_Resp_NPI()
rrLMS = Radar_Resp_LMS()

i=0
# rrNPI.start()
# rrLMS.start()
try:
    while i<len(sig):
        rrNPI.add_data(sig[i])
        rrLMS.add_data(sig[i])
        
        npi_sig.append(rrNPI.get_sub_point())
        lms_sig.append(rrLMS.get_sub_point())
        
        npi_mod.append(rrNPI.get_model_point())
        lms_mod.append(rrLMS.get_linear_point())
        
        # npeaks.append(rr.get_npeak())
        # corr.append(rr.get_correlation())
        # time.sleep(1/300000)  # Simulate real-time data feed
        i+=1
        
except KeyboardInterrupt:
    print("Stopping thread due to keyboard interruption.")

# rrNPI.stop()
# rrLMS.stop()
npeaks=[]
# nnpeaks = rr.get_npeaks()
# for peak in nnpeaks:
#     npeaks.append(peak)
#%% Plot
time_axis = np.arange(len(sig)) / 30  # Time in seconds
# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column
axes[0].set_title('Radar', fontsize=16)
axes[0].plot(time_axis,sig, label='Signal', color='red')
axes[0].plot(time_axis,npi_mod, label="NPI Model", linestyle='--', color='blue')
axes[0].plot(time_axis,lms_mod, label="LMS Model", linestyle='--', color='darkorange')
axes[0].grid()
axes[0].legend()
axes[0].set_ylabel('Displacement (unitless)')

# 2nd subplot: Signal and Linear Model
axes[1].plot(time_axis,npi_sig, label="Detrended NPI", color='blue')
axes[1].plot(time_axis,lms_sig, label="Detrended LMS", color='darkorange')
axes[1].set_ylabel('Displacement (unitless)')
axes[1].legend()
axes[1].grid()
axes[1].set_title('Subtract Model', fontsize=16)

# 3rd subplot: Pneum
axes[2].plot(time_axis,truth, label='Pneum', color='black')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Displacement (unitless)')
axes[2].legend()
axes[2].grid()
axes[2].set_title('Pneum', fontsize=16)

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
















class Radar_Resp_LMS():
    '''' Begin Class '''
    def __init__(self):
        self.fs = 30                    # frequency (Hz)
        
        self.win_size = 12 * self.fs     # window size (s)
        self.check_corr = 0.3 * self.fs   # rate to check if model still correlates to signal (s)
        self.corr_threshold = 0.7       # correlation threshold to refresh model
        self.window = []                # empty window
        
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
        if len(self.window) >= self.win_size:
            self.window.pop(0)
        self.window.append(value)
        if len(self.window) == self.win_size:
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
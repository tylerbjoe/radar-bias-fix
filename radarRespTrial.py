import time
import numpy as np
from scipy import integrate
from scipy import stats
from scipy.stats import pearsonr
from threading import Thread
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Radar_Resp():
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
    
#%% LEGACY
    # def set_peak_slope(self):
    #     last_peaks = find_peaks(self.window, distance=2*30, prominence=200)[0]#1650 prominence when breathold
    #     if len(last_peaks) >= 1:
    #         x2 = self.sample_n - (self.win_size - last_peaks[-1])
    #         y2 = self.window[last_peaks[-1]]
    #         if len(last_peaks) >= 2:
    #             x1 = self.sample_n - self.win_size + last_peaks[-2]
    #             y1 = self.window[last_peaks[-2]]
    #         else:
    #             x1 = self.lastnpeak[0]
    #             y1 = self.lastnpeak[1]
    #         self.lastnpeak=[x2,y2]
    #         self.slope = (y1-y2)/(x1-x2)

# %% Try on existing data
import json
import pandas as pd

# %% For JOSNS
# with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\testing_10_22 1\testing_10_22\inhale_hold\Radar_1_metadata_1729626016.450956.json", 'r') as file:
#     json_data = json.load(file)
# bins = []
# df = pd.DataFrame(json_data)
# for i in range(6):
#     tmp_b, tmp_a = [], []
#     for j in range(1, len(df["frame_data"])):
#         tmp_b.append(df["frame_data"][j][i])
#     bins.append(tmp_b)
# biny = [sum(values) for values in zip(bins[0],bins[1],bins[2],bins[3],bins[4],bins[5])]
# sig = integrate.cumulative_trapezoid(biny)

#%% csvs
file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_1\Pneumo.csv"
sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
truth = [int(x) for x in sigs][:15000]

file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data 1\Radar_Pneumo Data\Subject_1\Radar_2.csv"
sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:][:15000]
sig = [float(x) for x in sigs]
sig=np.array(sig)
#%%
# Example usage:
subtracted_sig = []
lin_mod = []
corr=[]
npeaks = []
slopes=[]

rr = Radar_Resp()

i=0
rr.start()
try:
    while i<len(sig):
        rr.add_data(sig[i])
        subtracted_sig.append(rr.get_sub_point())
        lin_mod.append(rr.get_model_point())
        slopes.append(rr.get_slope())
        # npeaks.append(rr.get_npeak())
        # corr.append(rr.get_correlation())
        time.sleep(1/300000)  # Simulate real-time data feed
        i+=1
        
except KeyboardInterrupt:
    print("Stopping thread due to keyboard interruption.")

rr.stop()
npeaks=[]
nnpeaks = rr.get_npeaks()
for peak in nnpeaks:
    npeaks.append(peak)

#%%
time_axis = np.arange(len(sig)) / 30  # Time in seconds
# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column

# First subplot: Signal and Linear Model
axes[0].plot(time_axis,sig, label='Signal', color='blue')
axes[0].scatter(time_axis[npeaks],sig[npeaks], label="Detected Negative Peak", color='orange')
axes[0].plot(time_axis,lin_mod, label="NPI Model", color='orange')
axes[0].plot(time_axis,subtracted_sig, label="Detrended (Signal-Model)", color='red')



# for peak in npeaks:
#     axes[0].scatter(time_axis[peak[0]],peak[1], color='orange')    

# axes[0].plot(time_axis, truth, label='Truth', color='black')
axes[0].set_ylabel('Displacement (unitless)')
axes[0].set_xlabel('Time')
axes[0].legend()
axes[0].grid()
axes[0].set_title('Radar', fontsize=16)

# axes[1].plot(time_axis,slopes, label="Slopes", color='magenta')
#%%
# Second subplot: Correlation
# axes[1].plot(time_axis, corr, label='Correlation', color='green')
axes[1].plot(time_axis,truth, label='Pneum', color='black')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Displacement (unitless)')
axes[1].legend()
axes[1].grid()
axes[1].set_title('Pneum', fontsize=16)

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
import time
import numpy as np
from scipy import integrate
from scipy import stats
from scipy.stats import pearsonr
from threading import Thread
import matplotlib.pyplot as plt

class Radar_Resp():
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



# %% Try on existing data
import json
import pandas as pd

# %% For JOSNS
with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\testing_10_22 1\testing_10_22\inhale_hold\Radar_1_metadata_1729626016.450956.json", 'r') as file:
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

# #%% csvs
# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data\Radar_Pneumo Data\Subject_2\Pneumo.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
# truth = [int(x) for x in sigs][:15000]

# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data\Radar_Pneumo Data\Subject_2\Radar_2.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:][:15000]
# sig = [float(x) for x in sigs]
#%%
# Example usage:
subtracted_sig = []
lin_mod = []
corr=[]
rr = Radar_Resp()
i=0
while i<300:
    rr.add_data(sig[i])
    subtracted_sig.append(0)
    lin_mod.append(0)
    corr.append(0)
    i+=1
rr.start()

try:
    while i<len(sig):
        rr.add_data(sig[i])
        subtracted_sig.append(rr.get_sub_point())
        lin_mod.append(rr.get_linear_point())
        corr.append(rr.get_correlation())
        time.sleep(1/30000)  # Simulate real-time data feed
        i+=1
        
except KeyboardInterrupt:
    print("Stopping thread due to keyboard interruption.")
    rr.stop()

rr.stop()

#%%
time_axis = np.arange(len(sig)) / 30  # Time in seconds
# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column

# First subplot: Signal and Linear Model
axes[0].plot(time_axis,sig, label='Signal', color='blue')#time_axis, 
axes[0].plot(time_axis,lin_mod, label="LMS Model", color='orange')
axes[0].plot(time_axis,subtracted_sig, label="Signal Linear Subtracted", color='red')
# axes[0].plot(time_axis, truth, label='Truth', color='black')
axes[0].set_ylabel('Displacement (unitless)')
axes[0].legend()
axes[0].grid()
axes[0].set_title('Radar', fontsize=16)
# #%%
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
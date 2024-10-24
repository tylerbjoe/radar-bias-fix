import time
import numpy as np
from scipy import integrate, stats
from scipy.stats import pearsonr
from threading import Thread
import matplotlib.pyplot as plt
from queue import Queue

class Radar_Resp():
    '''' Begin Class '''
    def __init__(self):
        self.window = []
        self.fs = 30
        self.win_size = 7 * self.fs
        self.check_corr = 7 * self.fs
        self.running = False
        self.lin_model = []
        self.sub_point = 0
        self.correlation = 0
        self.sample_n = 0
        self.slope = 0
        self.intercept = 0
        self.last_lin_model = []  # New variable to track last linear model

    def get_sub_point(self):
        return self.sub_point
    
    def set_sub_point(self):
        self.sub_point = self.window[-1] - self.get_linear_point()

    def set_correlation(self):
        self.correlation, _ = pearsonr(self.window, self.lin_model)
        
    def get_correlation(self):
        return self.correlation
    
    def add_data(self, value):
        self.sample_n += 1
        if len(self.window) >= self.win_size:
            self.window.pop(0)
        self.window.append(value)
        
        # Update linear model and correlation as new data is added
        if len(self.window) == self.win_size:
            if self.sample_n != self.win_size:
                self.set_correlation()
            if self.sample_n % self.check_corr == 0 and self.correlation <= 0.2:
                self.set_linear()
            self.set_sub_point()
        
    def set_linear(self):
        dur = len(self.window)
        x = np.linspace(0, dur / self.fs, dur)
        
        # If there's a previous linear model, use its last value to adjust the intercept
        if self.last_lin_model:
            last_point = self.last_lin_model[-1]
            self.slope, _, r, p, std_err = stats.linregress(x, self.window)
            # Adjust intercept to align with the last point of the previous model
            self.intercept = last_point - self.slope * (dur / self.fs)
        else:
            self.slope, self.intercept, r, p, std_err = stats.linregress(x, self.window)

        self.lin_model = self.slope * x + self.intercept
        self.last_lin_model = self.lin_model.copy()  # Update last linear model

    def get_linear_point(self):
        return self.slope * (self.sample_n / self.fs) + self.intercept

    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        try:
            while self.running:
                print(f"{self.sub_point} Linear subtracted point")
                time.sleep(1 / 3000)
        except:
            self.running = False

    def stop(self):
        self.running = False
    '''' End Class '''

# Try on existing data
import json
import pandas as pd
with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\test_npy_10_15 1\test_npy_10_15\Hunter\Hunter4\Radar_1_metadata_1729021313.5671923.json", 'r') as file:
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
        time.sleep(1/3000)  # Simulate real-time data feed
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
axes[0].plot(time_axis, sig, label='Signal', color='blue')
axes[0].plot(time_axis, lin_mod, label="LMS Model", color='orange')
axes[0].plot(time_axis, subtracted_sig, label="Signal Linear Subtracted", color='red')
axes[0].set_ylabel('Displacement (unitless)')
axes[0].legend()
axes[0].grid()
axes[0].set_title('Signal Analysis', fontsize=16)

# Second subplot: Correlation
axes[1].plot(time_axis, corr, label='Correlation', color='green')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Linear Correlation')
axes[1].legend()
axes[1].grid()
axes[1].set_title('Correlation Analysis', fontsize=16)

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
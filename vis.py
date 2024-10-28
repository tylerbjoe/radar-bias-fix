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




# windowing function takes array and returns array of arrays containing windows
def get_windows(signal, window_len=10, overlap=0, fz=30):
    windows = []
    win_dur = int(window_len * fz)
    win_diff = int(window_len-overlap * fz)
    #loop through sinal to get windows
    i = 0
    while i < len(signal):
        win = signal[i : i + win_dur]
        windows.append(win)
        i += win_dur
    return windows

# linear model fit to window using least squares
def get_linear(signal):
    dur = len(signal)
    x = np.linspace(0, dur/30, dur)
    slope, intercept, r, p, std_err = stats.linregress(x, signal)
    return slope * x + intercept
   
def get_peakline(signal):
    last_peaks = find_peaks(signal, distance=2.5*30, prominence=1650)[0]
    x1 = last_peaks[-2]
    y1 = signal[x1]
    x2 = last_peaks[-1]
    y2 = signal[x2]
    signal = (y2-y1)/(x2-x1)
    return last_peaks

#%% JSONS
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
#%% CSVS
# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data\Radar_Pneumo Data\Subject_2\Pneumo.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:]
# truth = [int(x) for x in sigs][:15000]

# file_path = r"C:\Users\TJoe\Documents\Radar Offset Fix\Radar_Pneumo Data\Radar_Pneumo Data\Subject_2\Radar_2.csv"
# sigs = pd.read_csv(file_path, usecols=[0], header=None).squeeze().tolist()[1:][:15000]
# sig = [float(x) for x in sigs]
# sig = np.array(sig)

# # script to run
# fs = 18

#%% WINDOWING
windows = get_windows(sig)
window=[]
lin=[]
time=[]
subtracted=[]
for win in windows:
    for val in win:
        window.append(val)
    lin_win = get_linear(win)
    for l in lin_win:
        lin.append(l)
    times = (np.arange(start=len(time), stop=len(window))/30)
    for t in times:
        time.append(t)
i=0
while i<len(window):
    subtracted.append(window[i]-lin[i])
    i+=1

#%% SIGNAL PROCESSING
peaks = get_peakline(-sig)
time = np.array(time)
sig = np.array(sig)

negative_peak_values = sig[peaks]

interpolator = interp1d(peaks, negative_peak_values, kind='linear', fill_value="extrapolate")
interpolated_signal = interpolator(np.arange(len(sig)))#[peaks[0]:peaks[-1]])))

#%% PLOTTING SINGLE
fs=18
plt.figure()
# plt.plot(time,window, label='signal')
# plt.plot(time,lin, label="lms model")
# plt.plot(time,subtracted, label="signal linear subtracted")
plt.plot(time,sig, label='Signal',color='blue')
plt.scatter(time[peaks],sig[peaks], color='orange')
plt.plot(time,interpolated_signal, label='Interpolated Negative Peaks', linestyle='--', color='orange')#[peaks[0]:peaks[-1]]
plt.plot(time,sig-interpolated_signal, label='detrended signal', color='red')

#plt.title('8 Second window', fontsize=fs)
plt.ylabel('displacement', fontsize=fs)
plt.xlabel('time (s)', fontsize=fs)
plt.legend(fontsize=fs)
plt.grid()
plt.show()
   
#%% SUBPLOTS W TRUTH
# time_axis = time
# # Create subplots
# fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column

# # First subplot: Signal and Linear Model
# axes[0].plot(time_axis,sig, label='Signal', color='blue')#time_axis, 
# axes[0].plot(time_axis[peaks[0]:peaks[-1]],interpolated_signal, label='Interpolated Negative Peaks', linestyle='--', color='orange')
# axes[0].plot(time_axis[peaks[0]:peaks[-1]],sig[peaks[0]:peaks[-1]]-interpolated_signal, color='red')

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
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
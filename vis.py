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

with open(r"C:\Users\TJoe\Documents\Radar Offset Fix\test_npy_10_15 1\test_npy_10_15\Hunter\Hunter4\Radar_1_metadata_1729021313.5671923.json", 'r') as file:
    json_data = json.load(file)

bins = []

df = pd.DataFrame(json_data)
for i in range(6):
    tmp_b, tmp_a = [], []
    for j in range(1, len(df["frame_data"])):
        tmp_b.append(df["frame_data"][j][i])
    bins.append(tmp_b)

# x_axis = df["frame_times"]
# x_axis = np.array(x_axis - (min(x_axis)))

# fig, axes = plt.subplots(3, 1, figsize=(15, 8))
# bin_plots = [0]*6
# axes[0].grid()
# axes[0].set_xlabel('Time(s)')
# axes[0].set_ylabel('Velocity(unitless)')
# axes[1].grid()
# axes[1].set_xlabel('Time(s)')
# axes[1].set_ylabel('Displacement(unitless)')
# for i in range(6):
#     bin_plots[i], = axes[0].plot(x_axis[:len(bins[i])], bins[i])
#     bin_plots[i], = axes[1].plot(x_axis[:len(bins[i])-1], integrate.cumulative_trapezoid(bins[i]))
# plt.show()


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
   
   
   
   
   
# script to run
fs = 18

biny = [sum(values) for values in zip(bins[0],bins[1],bins[2],bins[3],bins[4],bins[5])]
sig = integrate.cumulative_trapezoid(biny)
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


plt.figure()
plt.plot(time,window, label='signal')
plt.plot(time,lin, label="lms model")
plt.plot(time,subtracted, label="signal linear subtracted")

#plt.title('8 Second window', fontsize=fs)
plt.ylabel('displacement', fontsize=fs)
plt.xlabel('time (s)', fontsize=fs)
plt.legend(fontsize=fs)
plt.show()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
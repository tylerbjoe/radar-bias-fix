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
        self.window = []
        self.fs = 30
        self.win_size = 10 * self.fs
        self.running = False
        self.lin_model=[]
        self.sub_point=0
        self.correlation = 0

    def get_sub_point(self):
        return self.sub_point
    
    def set_sub_point(self):
        if len(self.window) == self.win_size and self.correlation>0.5:
            self.sub_point = self.lin_model[-1] - self.window[-1]
        elif len(self.window) == self.win_size and self.correlation<=0.5:
            self.get_linear()
            self.sub_point = self.lin_model[-1] - self.window[-1]
            
            
    def set_correlation(self):
        if len(self.window) == len(self.lin_model):
            self.correlation,_ = pearsonr(self.window, self.lin_model)
        
    # def get_correlation(self):
    #     return self.correlation
    
    def add_data(self, value):
        if len(self.window) >= self.win_size:
            self.window.pop(0)
        self.window.append(value)
        
        # Update linear model and correlation as new data is added
        if len(self.window) == self.win_size:
            self.get_linear()
            self.set_correlation()
            self.set_sub_point()
        
    def get_linear(self):
        dur = len(self.window)
        x = np.linspace(0, dur/30, dur)
        slope, intercept, r, p, std_err = stats.linregress(x, self.window)
        self.lin_model = slope * x + intercept
        
    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        try:
            while self.running:
                # Print the subtracted point at regular intervals
                print(f"{self.sub_point} Linear subtracted point")
                time.sleep(1)
        except KeyboardInterrupt:
            # Stop the thread if interrupted by keyboard (e.g., in Jupyter)
            self.running = False

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to stop fully
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
rr = Radar_Resp()
for point in sig[:300]:
    rr.add_data(point)
rr.start()

#use signal
for point in sig[300:]:
    rr.add_data(point)
    time.sleep(1)

rr.stop()
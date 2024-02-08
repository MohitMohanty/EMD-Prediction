


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import threading
import nvidia_smi

# Function to fetch GPU utilization in a separate thread
def get_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        print(f"|Device {i}| Mem Free: {mem.free/1024**2:.2f}MB / {mem.total/1024**2:.2f}MB | gpu-util: {util.gpu/100.0:.1%} | gpu-mem: {util.memory/100.0:.1%} |")

        # Adjust the sleep time based on how often you want to fetch GPU utilization
        # For example, every 5 seconds:
        threading.Timer(5.0, get_gpu_utilization).start()

# Start the GPU utilization thread
get_gpu_utilization()

# The rest of your existing code...
# ... (your existing code remains unchanged)

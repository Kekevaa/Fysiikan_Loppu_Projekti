
import streamlit as st
import pandas as pd 
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt

path1 = "Location.csv"
df = pd.read_csv(path1)

path2 = "Linear Accelerometer.csv"
df2 = pd.read_csv(path2)

st.title('My short walk data')

from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, fs, nyq, order): 
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, nyq, order): 
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

T = df2['Time (s)'][len(df2['Time (s)'])-1] - df2['Time (s)'][0]
n = len(df2['Time (s)'])
fs = n/T
nyq = fs/2
order = 3
cutoff = 1/(0.2)

df2['filter_a_z'] = butter_lowpass_filter( df2['Z (m/s^2)'], cutoff, fs, nyq, order)

from scipy.signal import find_peaks

peaks, _ = find_peaks(df2['filter_a_z'], distance=fs*0.2) 

valleys, _ = find_peaks(-df2['filter_a_z'], distance=fs*0.2)
total_steps = min(len(peaks), len(valleys))

st.write("Step count calculated using filtering:",total_steps, "steps")


time = df2.iloc[:, 0].values
signal = df2.iloc[:, 1].values

dt = time[1] - time[0]

N = len(signal)  
fourier = np.fft.fft(signal, N) 
psd = (fourier * np.conj(fourier)) / N  
freq = np.fft.fftfreq(N, dt)

step_freq_range = (1, 3) 
step_freqs = np.logical_and(freq > step_freq_range[0], freq < step_freq_range[1])

step_psd = psd[step_freqs]
step_count = len(step_psd) 

st.write("Step count estimated based on Fourier analysis:",step_count, "steps")


R = 6371000.0  

def haversine(lat1, lon1, lat2, lon2):
  
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c 
    return distance

total_distance = 0
for i in range(1, len(df)):
    lat1, lon1 = df.iloc[i-1]['Latitude (°)'], df.iloc[i-1]['Longitude (°)']
    lat2, lon2 = df.iloc[i]['Latitude (°)'], df.iloc[i]['Longitude (°)']
    total_distance += haversine(lat1, lon1, lat2, lon2)

total_distance_rounded = round(total_distance, 1)
st.write("Total distance: ", total_distance_rounded, 'm')

total_time = df['Time (s)'].iloc[-1] - df['Time (s)'].iloc[0]  
total_minutes = int(total_time // 60)
total_seconds = int(total_time % 60)

st.write("Total time:",total_minutes, "min",total_seconds,"s")


average_speed = total_distance / total_time  
average_speed_rounded = round(average_speed,2)
                              
st.write("Average speed: ", average_speed_rounded, "m/s")



step_length = total_distance / total_steps 
step_length_rounded = round(step_length, 2)
st.write("Step length:", step_length_rounded, "m")



st.title('Filtered acceleration data.')
df2['Time (min)'] = df2['Time (s)'] / 60
plt.figure(figsize=(25, 10))
plt.plot(df2['Time (min)'], df2['filter_a_z'], label='Filtered Z-component')
plt.xlabel('Time (t)')
plt.ylabel('Acceleration (m/s^2)')

st.pyplot(plt)

st.title('Filtered acceleration data in the interval 0.4-2 ')
df2['Time (min)'] = df2['Time (s)'] / 60

plt.figure(figsize=(18, 10))
plt.xlim(0.4, 2)
plt.plot(df2['Time (min)'], df2['filter_a_z'], label='Filtered Z-component')
plt.xlabel('Time (t)')
plt.ylabel('Acceleration (m/s^2)')

st.pyplot(plt)


from scipy.fft import fft, fftfreq
st.title('Power spectrum')


def plot_power_spectrum(signal, fs, label):
    N = len(signal)
    fourier = fft(signal)
    freqs = fftfreq(N, 1/fs)
    psd = np.abs(fourier)**2 / N
    pos_freqs = freqs[:N//2]
    pos_psd = psd[:N//2]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pos_freqs, pos_psd, label=label)
    ax.set_xlim(0, 20)
    ax.set_xlabel('Frequenzy (Hz)')
    ax.set_ylabel('Power spectral density (dB)')
    ax.legend()
    
    return fig

if 'filter_a_z' in df2.columns:
    dt = df2['Time (s)'].iloc[1] - df2['Time (s)'].iloc[0]  
    fs = 1 / dt  
    fig = plot_power_spectrum(df2['filter_a_z'], fs, 'Filtered Z-component')
    st.pyplot(fig)
else:
    st.error("The DataFrame does not contain the 'filter_a_z' column.")

st.title('Traveled route on the map')
start_lat = df['Latitude (°)'].mean()
start_long = df['Longitude (°)'].mean()

map = folium.Map(location=[start_lat, start_long], zoom_start=16, 
                 scrollWheelZoom=False, 
                 dragging=False) 

folium.PolyLine(df[['Latitude (°)', 'Longitude (°)']], color='red', weight=2.5, opacity=1).add_to(map)
st_map = st_folium(map, width=900, height=650)

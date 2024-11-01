import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from scipy.integrate import simpson as simps


module_name = "utils_v2"
file_path = "/workspaces/TP-Chaos/utils_v2.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Importer le module
import utils_v2 as u

import pandas as pd


#sampling period
dt =0.002



#Import the data
#----config1-----#
C1 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_1.txt",usecols=[0,1,2],skiprows=6, delimiter='\t', decimal=',', encoding='ISO-8859-1')

C1_t = C1.iloc[:,0]
C1_theta = C1.iloc[:,1]
C1_thetadot = C1.iloc[:,2]

#----config2-----#
#3 périodes mieux
C2_1 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_2_1.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C2_1_t=C2_1.iloc[:,0]
C2_1_theta=C2_1.iloc[:,1]
C2_1_thetadot=C2_1.iloc[:,2]

#3 pértiodes inversé
C2_2 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_2_2.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C2_2_t=C2_2.iloc[:,0]
C2_2_theta=C2_2.iloc[:,1]
C2_2_thetadot=C2_2.iloc[:,2]

#1 période
C2_3 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_2_4.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C2_3_t=C2_3.iloc[:,0]
C2_3_theta=C2_3.iloc[:,1]
C2_3_thetadot=C2_3.iloc[:,2]

#----config3-----#
C3_1 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_3_1.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C3_1_t=C3_1.iloc[:,0]
C3_1_theta=C3_1.iloc[:,1]
C3_1_thetadot=C3_1.iloc[:,2]

C3_2 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_3_2.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C3_2_t=C3_2.iloc[:,0]
C3_2_theta=C3_2.iloc[:,1]
C3_2_thetadot=C3_2.iloc[:,2]

C3_3 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_3_3.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C3_3_t=C3_3.iloc[:,0]
C3_3_theta=C3_3.iloc[:,1]
C3_3_thetadot=C3_3.iloc[:,2]

#----config4-----#
#1 période
C4 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas_Moteur/Config_4.txt", skiprows=6, usecols=[0, 1, 2], delimiter='\t', decimal = ",", encoding='ISO-8859-1')  # Adjust delimiter if necessary
C4_t=C4.iloc[:,0]
C4_theta=C4.iloc[:,1]
C4_thetadot=C4.iloc[:,2]



#-----------config 1------------#

#----plot----#
t = C1_t
theta = C1_theta
thetadot = C1_thetadot

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.2,0.4])
ax.set_ylim([-0.1,0.1])

# ax.plot(theta,thetadot,color="black",linestyle=':', linewidth=0.5)
ax.plot(theta,thetadot,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config1_phase_space.pdf")

#theta
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t,theta,color="black")
time_range = [0, 4]
ax.set_xlim(time_range)

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config1_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config1_thetadot.pdf")

#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

norm = simps(positive_fft, x =positive_freqs)
positive_fft = positive_fft/norm

max_freq = positive_freqs[np.argmax(positive_fft)]

#plot the harmonics
#we want k to be half integer, i.e k= -1/2,1/2,3/2,5/2,7/2,9/2,11/2,13/2,15/2,...add()
k = np.arange(-15,15)
k = k/2
k = k[k!=1]
ax.vlines(k*max_freq, 0, 4, color="blue", linestyle="-.", label = r"$x = \frac{k}{2}\nu^\star$, k $\in \mathbb{Z}$")


print("Max frequency, C1: ",max_freq)
#plot the max frequency
ax.axvline(x=max_freq, color="red", linestyle="--",label = rf"$x = \nu^\star$ = {max_freq:.2f} Hz")

#plot spectral analysis
ax.plot(positive_freqs, positive_fft, color="black")


freq_range = [0, 20]
ax.set_xlim(freq_range)
ax.set_ylim([0, 3.4])
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config1_spectral_analysis.pdf")


#make the poincaré section at t = n*T with T the excitation period, i.e. 1/max_freq
exc_freq = 4
T = 1/exc_freq
n = 3
t_poincare = t[t % T < dt]
theta_poincare_1 = theta[t % T < dt]
thetadot_poincare_1 = thetadot[t % T < dt]

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.2,0.4])
ax.set_ylim([-0.1,0.1])

ax.scatter(theta_poincare_1,thetadot_poincare_1,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config1_poincare.pdf")


# #-----------config 2_1------------#

# #----plot----#
# t = C2_1_t
# theta = C2_1_theta
# thetadot = C2_1_thetadot

# #phase space
# xlabel = r"$\theta$" + " [V]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(theta,thetadot,color="black")

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_1_phase_space.pdf")

# #theta
# xlabel = "Time [s]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,theta,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_1_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_1_thetadot.pdf")

# #spectral analysis
# xlabel = "Frequency [Hz]"
# ylabel = "Amplitude"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# fs = 1/(t[1]-t[0])
# n = len(theta)

# #FFT
# theta_demeaned = theta - np.mean(theta)
# theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
# n = len(theta_demeaned)
# fft_values = np.fft.fft(theta_demeaned)
# frequencies = np.fft.fftfreq(n, d=1/fs)
# positive_freqs = frequencies[:n//2]
# positive_fft = np.abs(fft_values[:n//2])

# ax.plot(positive_freqs, positive_fft, color="black")

# freq_range = [0, 30]
# ax.set_xlim(freq_range)
# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_1_spectral_analysis.pdf")


# #-----------config 2_2------------#

# #----plot----#
# t = C2_2_t
# theta = C2_2_theta
# thetadot = C2_2_thetadot

# #phase space
# xlabel = r"$\theta$" + " [V]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(theta,thetadot,color="black")

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_2_phase_space.pdf")

# #theta
# xlabel = "Time [s]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,theta,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_2_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_2_thetadot.pdf")

# #spectral analysis
# xlabel = "Frequency [Hz]"
# ylabel = "Amplitude"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# fs = 1/(t[1]-t[0])
# n = len(theta)

# #FFT
# theta_demeaned = theta - np.mean(theta)
# theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
# n = len(theta_demeaned)
# fft_values = np.fft.fft(theta_demeaned)
# frequencies = np.fft.fftfreq(n, d=1/fs)
# positive_freqs = frequencies[:n//2]
# positive_fft = np.abs(fft_values[:n//2])

# ax.plot(positive_freqs, positive_fft, color="black")

# freq_range = [0, 30]
# ax.set_xlim(freq_range)
# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_2_spectral_analysis.pdf")

#-----------config 2_3------------#

#----plot----#
t = C2_3_t
theta = C2_3_theta
thetadot = C2_3_thetadot

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(theta,thetadot,color="black")

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.1,0.1])

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_3_phase_space.pdf")

#theta
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t,theta,color="black")
time_range = [0.25, 3.25]
ax.set_xlim(time_range)

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_3_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_3_thetadot.pdf")

#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

norm = simps(positive_fft, x =positive_freqs)
positive_fft = positive_fft/norm

max_freq = positive_freqs[np.argmax(positive_fft)]

#plot the harmonics
#we want k to be half integer, i.e k= -1/2,1/2,3/2,5/2,7/2,9/2,11/2,13/2,15/2,...add()
k = np.arange(-15,15)
k = k/3
k = k[k!=1]
ax.vlines(k*max_freq, 0, 4, color="blue", linestyle="-.", label = r"$x = \frac{k}{3}\nu^\star$, k $\in \mathbb{Z}$")


print("Max frequency, C2_3: ",max_freq)
#plot the max frequency
ax.axvline(x=max_freq, color="red", linestyle="--",label = rf"$x = \nu^\star$ = {max_freq:.2f} Hz")

ax.plot(positive_freqs, positive_fft, color="black")

freq_range = [0, 15]
ax.set_xlim(freq_range)
ax.set_ylim([0, 2.25])
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_3_spectral_analysis.pdf")


#make the poincaré section at t = n*T with T the excitation period, i.e. 1/max_freq
exc_freq = 3
T = 1/exc_freq
t_poincare = t[t % T < dt]
theta_poincare_2 = theta[t % T < dt]
thetadot_poincare_2 = thetadot[t % T < dt]

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.scatter(theta_poincare_2,thetadot_poincare_2,color="black")

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.1,0.1])

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config2_3_poincare.pdf")

#-----------config 3_1------------#

#----plot----#
t = C3_1_t
theta = C3_1_theta
thetadot = C3_1_thetadot

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(theta,thetadot,color="black")

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.08,0.08])

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_1_phase_space.pdf")

#theta
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t,theta,color="black")
time_range = [0, 4]
ax.set_xlim(time_range)

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_1_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_1_thetadot.pdf")

#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

norm = simps(positive_fft, x =positive_freqs)
positive_fft = positive_fft/norm

max_freq = positive_freqs[np.argmax(positive_fft)]

#plot the harmonics
#we want k to be half integer, i.e k= -1/2,1/2,3/2,5/2,7/2,9/2,11/2,13/2,15/2,...add()
k = np.arange(-100,100)
k = k/5
k = k[k!=1]
ax.vlines(k*max_freq, 0, 4, color="blue", linestyle="-.", label = r"$x = \frac{k}{5}\nu^\star$, k $\in \mathbb{Z}$")

print("Max frequency, C3_1: ",max_freq)
#plot the max frequency
ax.axvline(x=max_freq, color="red", linestyle="--",label = rf"$x = \nu^\star$ = {max_freq:.2f} Hz")

ax.plot(positive_freqs, positive_fft, color="black")

freq_range = [0, 15]
ax.set_xlim(freq_range)
ax.set_ylim([0, 1.5])
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_1_spectral_analysis.pdf")


#make the poincaré section at t = n*T with T the excitation period, i.e. 1/max_freq
exc_freq = 2.5
T = 1/exc_freq
t_poincare = t[t % T < dt]
theta_poincare_3_1 = theta[t % T < dt]
thetadot_poincare_3_1 = thetadot[t % T < dt]

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.08,0.08])

ax.scatter(theta_poincare_3_1,thetadot_poincare_3_1,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_1_poincare.pdf")


#-----------config 3_2------------#

#----plot----#
t = C3_2_t
theta = C3_2_theta
thetadot = C3_2_thetadot

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.08,0.08])

ax.plot(theta,thetadot,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_2_phase_space.pdf")

#theta
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t,theta,color="black")
time_range = [0.5, 4.5]
ax.set_xlim(time_range)
ax.set_ylim([-0.3, 0.4])


u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_2_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0.35, 4.35]
# ax.set_xlim(time_range)


# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_2_thetadot.pdf")

#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])


norm = simps(positive_fft, x =positive_freqs)
positive_fft = positive_fft/norm

max_freq = positive_freqs[np.argmax(positive_fft)]

#plot the harmonics
#we want k to be half integer, i.e k= -1/2,1/2,3/2,5/2,7/2,9/2,11/2,13/2,15/2,...add()
k = np.arange(-100,100)
k = k/5
k = k[k!=1]
ax.vlines(k*max_freq, 0, 4, color="blue", linestyle="-.", label = r"$x = \frac{k}{5}\nu^\star$, k $\in \mathbb{Z}$")


print("Max frequency, C3_2: ",max_freq)
#plot the max frequency
ax.axvline(x=max_freq, color="red", linestyle="--",label = rf"$x = \nu^\star$ = {max_freq:.2f} Hz")


ax.plot(positive_freqs, positive_fft, color="black")

freq_range = [0, 15]
ax.set_xlim(freq_range)
ax.set_ylim([0, 1.5])
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_2_spectral_analysis.pdf")

#zoomed version of the spectral analysis
freq_range = [0, 8]
ax.set_xlim(freq_range)
ax.set_ylim([0, 0.5])
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_2_spectral_analysis_ZOOM.pdf")




#make the poincaré section at t = n*T with T the excitation period, i.e. 1/max_freq
exc_freq = 2.5
T = 1/exc_freq
n = 3
t_poincare = t[t % T < dt]
theta_poincare_3_2 = theta[t % T < dt]
thetadot_poincare_3_2 = thetadot[t % T < dt]

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.08,0.08])

ax.scatter(theta_poincare_3_2,thetadot_poincare_3_2,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_2_poincare.pdf")

#-----------config 3_3------------#

#----plot----#
t = C3_3_t
theta = C3_3_theta
thetadot = C3_3_thetadot

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.3,0.45])
ax.set_ylim([-0.12,0.12])

ax.plot(theta,thetadot,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_3_phase_space.pdf")

#theta
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t,theta,color="black")
time_range = [0, 4]
ax.set_xlim(time_range)

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_3_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_3_thetadot.pdf")

#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

ax.plot(positive_freqs, positive_fft, color="black")

freq_range = [0, 30]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_3_spectral_analysis.pdf")


#make the poincaré section at t = n*T with T the excitation period, i.e. 1/max_freq
exc_freq = 2.5
T = 1/exc_freq
n = 3
t_poincare = t[t % T < dt]
theta_poincare_3_3 = theta[t % T < dt]
thetadot_poincare_3_3 = thetadot[t % T < dt]

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.3,0.45])
ax.set_ylim([-0.12,0.12])

ax.scatter(theta_poincare_3_3,thetadot_poincare_3_3,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config3_3_poincare.pdf")

#-----------config 4------------#

#----plot----#
t = C4_t
theta = C4_theta
thetadot = C4_thetadot


#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.14,0.12])

ax.plot(theta,thetadot,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config4_phase_space.pdf")

#theta
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t,theta,color="black")
time_range = [0, 4]
ax.set_xlim(time_range)

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config4_theta.pdf")

# #thetadot
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t,thetadot,color="black")
# time_range = [0, 4]
# ax.set_xlim(time_range)

# u.set_legend_properties(ax,fontsize=25)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config4_thetadot.pdf")

#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(t[1]-t[0])
n = len(theta)

#FFT
theta_demeaned = theta - np.mean(theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

norm = simps(positive_fft, x =positive_freqs)
positive_fft = positive_fft/norm

max_freq = positive_freqs[np.argmax(positive_fft)]

#plot the harmonics
#we want k to be half integer, i.e k= -1/2,1/2,3/2,5/2,7/2,9/2,11/2,13/2,15/2,...add()
k = np.arange(-100,100)
k = k
k = k[k!=1]
ax.vlines(k*max_freq, 0, 4, color="blue", linestyle="-.", label = r"$x = k\nu^\star$, k $\in \mathbb{Z}$")


print("Max frequency, C4: ",max_freq)
#plot the max frequency
ax.axvline(x=max_freq, color="red", linestyle="--",label = rf"$x = \nu^\star$ = {max_freq:.2f} Hz")

ax.plot(positive_freqs, positive_fft, color="black")

freq_range = [0, 30]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config4_spectral_analysis.pdf")


#make the poincaré section at t = n*T with T the excitation period, i.e. 1/max_freq
exc_freq = 7
T = 1/exc_freq
n = 3
t_poincare = t[t % T < dt]
theta_poincare_4 = theta[t % T < dt]
thetadot_poincare_4 = thetadot[t % T < dt]

#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)


ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.14,0.12])

ax.scatter(theta_poincare_4,thetadot_poincare_4,color="black")

u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_Config4_poincare.pdf")



#plot every poincaré section in the same plot
#phase space
xlabel = r"$\theta$" + " [V]"
ylabel = r"$\dot{\theta}$" + " [V]"

ax,fig = u.create_figure_and_apply_format((12,6),xlabel=xlabel, ylabel=ylabel)

ax.set_ylim([-0.125,0.13])
ax.set_xlim([-0.3,0.6])

ax.scatter(theta_poincare_4,thetadot_poincare_4,color="orange",label="1-periodic",marker = "s", s=60)
ax.scatter(theta_poincare_1,thetadot_poincare_1,color="blue",label="2-periodic",marker = "d", s=60)
ax.scatter(theta_poincare_2,thetadot_poincare_2,color="red",label="3-periodic",marker = "^", s=60)
ax.scatter(theta_poincare_3_2,thetadot_poincare_3_2,color="green",label="5-periodic",marker = "D", s=60)
ax.scatter(theta_poincare_3_3,thetadot_poincare_3_3,color="purple",label="Chaotic",marker = "H", s=60)



u.set_legend_properties(ax,fontsize=25)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/DP_poincare_all.pdf")


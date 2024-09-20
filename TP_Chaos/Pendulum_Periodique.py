import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os

module_name = "utils_v2"
file_path = "/workspaces/TP3/utils_v2.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Importer le module
import utils_v2 as u

import pandas as pd




#Import the data
C7_Droite = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config7_Droite.csv", delimiter=';', decimal=',')
C7_Droite_t=C7_Droite.iloc[:,0]
C7_Droite_theta=C7_Droite.iloc[:,1]
C7_Droite_thetadot=C7_Droite.iloc[:,2]

C7_Gauche = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config7_Gauche.csv", delimiter=';', decimal=',')
C7_Gauche_t=C7_Gauche.iloc[:,0]
C7_Gauche_theta=C7_Gauche.iloc[:,1]
C7_Gauche_thetadot=C7_Gauche.iloc[:,2]

C7_GaucheDroite = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config7_GaucheDroite.csv", delimiter=';', decimal=',')
C7_GaucheDroite_t=C7_GaucheDroite.iloc[:,0]
C7_GaucheDroite_theta=C7_GaucheDroite.iloc[:,1]
C7_GaucheDroite_thetadot=C7_GaucheDroite.iloc[:,2]

C8_C1_S1 = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config8_Cond1_Simul1.csv", delimiter=';', decimal=',')
C8_C1_S1_t=C8_C1_S1.iloc[:,0]
C8_C1_S1_theta=C8_C1_S1.iloc[:,1]
C8_C1_S1_thetadot=C8_C1_S1.iloc[:,2]

C8_C1_S2 = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config8_Cond1_Simul2.csv", delimiter=';', decimal=',')
C8_C1_S2_t=C8_C1_S2.iloc[:,0]
C8_C1_S2_theta=C8_C1_S2.iloc[:,1]
C8_C1_S2_thetadot=C8_C1_S2.iloc[:,2]

C8_C2_S1 = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config8_Cond2_Simul1.csv", delimiter=';', decimal=',')
C8_C2_S1_t=C8_C2_S1.iloc[:,0]
C8_C2_S1_theta=C8_C2_S1.iloc[:,1]
C8_C2_S1_thetadot=C8_C2_S1.iloc[:,2]

C8_C3_S1 = pd.read_csv("/workspaces/TP3/TP_Chaos/Datas/Config8_Cond3_Simul1.csv", delimiter=';', decimal=',')
C8_C3_S1_t=C8_C3_S1.iloc[:,0]
C8_C3_S1_theta=C8_C3_S1.iloc[:,1]
C8_C3_S1_thetadot=C8_C3_S1.iloc[:,2]


#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C7_Droite_t,C7_Droite_theta, label="Droite", color="blue")
ax.plot(C7_Gauche_t,C7_Gauche_theta, label="Gauche", color="red")
ax.plot(C7_GaucheDroite_t,C7_GaucheDroite_theta, label="GaucheDroite", color="green")

timerange = [-2,20]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config7_position.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [rad]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C7_Droite_t,C7_Droite_thetadot, label="Droite", color="blue")
ax.plot(C7_Gauche_t,C7_Gauche_thetadot, label="Gauche", color="red")
ax.plot(C7_GaucheDroite_t,C7_GaucheDroite_thetadot, label="GaucheDroite", color="green")

timerange = [-2,20]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config7_angul_speed.pdf")


#phase space
xlabel = r"$\dot{\theta}$" + " [rad]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C7_Droite_thetadot,C7_Droite_theta, label="Droite", color="blue")
ax.plot(C7_Gauche_thetadot,C7_Gauche_theta, label="Gauche", color="red")
ax.plot(C7_GaucheDroite_thetadot,C7_GaucheDroite_theta, label="GaucheDroite", color="green")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config7_phase_space.pdf")


#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(C7_Droite_t[1]-C7_Droite_t[0])

n1 = len(C7_Droite_theta)
n2 = len(C7_Gauche_theta)
n3 = len(C7_GaucheDroite_theta)

#FFT
theta_demeaned1 =  C7_Droite_theta - np.mean(C7_Droite_theta)
fft_values1 = np.fft.fft(theta_demeaned1)
frequencies1 = np.fft.fftfreq(n1, d=1/fs)

positive_freqs1 = frequencies1[:n1//2]
positive_fft1 = np.abs(fft_values1[:n1//2])

theta_demeaned2 = C7_Gauche_theta - np.mean(C7_Gauche_theta)
fft_values2 = np.fft.fft(theta_demeaned2)
frequencies2 = np.fft.fftfreq(n2, d=1/fs)

positive_freqs2 = frequencies2[:n2//2]
positive_fft2 = np.abs(fft_values2[:n2//2])

theta_demeaned3 = C7_GaucheDroite_theta - np.mean(C7_GaucheDroite_theta)
fft_values3 = np.fft.fft(theta_demeaned3)
frequencies3 = np.fft.fftfreq(n3, d=1/fs)

positive_freqs3 = frequencies3[:n3//2]
positive_fft3 = np.abs(fft_values3[:n3//2])

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")
ax.plot(positive_freqs3, positive_fft3, color="green",label="Exp. 3")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config7_spectral_analysis.pdf")

#Config8Cond1
#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_t,C8_C1_S1_theta, label="Cond1_Simul1", color="blue")
ax.plot(C8_C1_S2_t,C8_C1_S2_theta, label="Cond1_Simul2", color="red")

timerange = [-2,40]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond1_position.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [rad]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_t,C8_C1_S1_thetadot, label="Cond1_Simul1", color="blue")
ax.plot(C8_C1_S2_t,C8_C1_S2_thetadot, label="Cond1_Simul2", color="red")

timerange = [-2,40]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond1_angul_speed.pdf")


#phase space
xlabel = r"$\dot{\theta}$" + " [rad]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_thetadot,C8_C1_S1_theta, label="Cond1_Simul1", color="blue")
ax.plot(C8_C1_S2_thetadot,C8_C1_S2_theta, label="Cond1_Simul2", color="red")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond1_phase_space.pdf")


#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(C8_C1_S1_t[1]-C8_C1_S1_t[0])

n1 = len(C8_C1_S1_theta)
n2 = len(C8_C1_S2_theta)


#FFT
theta_demeaned1 = C8_C1_S1_theta - np.mean(C8_C1_S1_theta)
fft_values1 = np.fft.fft(theta_demeaned1)
frequencies1 = np.fft.fftfreq(n1, d=1/fs)

positive_freqs1 = frequencies1[:n1//2]
positive_fft1 = np.abs(fft_values1[:n1//2])

theta_demeaned2 = C8_C1_S2_theta - np.mean(C8_C1_S2_theta)
fft_values2 = np.fft.fft(theta_demeaned2)
frequencies2 = np.fft.fftfreq(n2, d=1/fs)

positive_freqs2 = frequencies2[:n2//2]
positive_fft2 = np.abs(fft_values2[:n2//2])

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond1_spectral_analysis.pdf")

#Config8Cond2and3
#position 

xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_t,C8_C1_S1_theta, label="Cond1_Simul1", color="blue")
ax.plot(C8_C2_S1_t,C8_C2_S1_theta, label="Cond2_Simul1", color="red")
ax.plot(C8_C3_S1_t,C8_C3_S1_theta, label="Cond3_Simul1", color="green")

timerange = [-2,120]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond2and3_position.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [rad]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_t,C8_C1_S1_thetadot, label="Cond1_Simul1", color="blue")
ax.plot(C8_C2_S1_t,C8_C2_S1_thetadot, label="Cond2_Simul1", color="red")
ax.plot(C8_C3_S1_t,C8_C3_S1_thetadot, label="Cond3_Simul1", color="green")

timerange = [-2,120]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond2and3_angul_speed.pdf")


#phase space
xlabel = r"$\dot{\theta}$" + " [rad]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_thetadot,C8_C1_S1_theta, label="Cond1_Simul1", color="blue")
ax.plot(C8_C2_S1_thetadot,C8_C2_S1_theta, label="Cond2_Simul1", color="red")
ax.plot(C8_C3_S1_thetadot,C8_C3_S1_theta, label="Cond3_Simul1", color="green")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond2and3_phase_space.pdf")


#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(C8_C1_S1_t[1]-C8_C1_S1_t[0])

n1 = len(C8_C1_S1_theta)
n2 = len(C8_C2_S1_theta)
n3 = len(C8_C3_S1_theta)


#FFT
theta_demeaned1 = C8_C1_S1_theta - np.mean(C8_C1_S1_theta)
fft_values1 = np.fft.fft(theta_demeaned1)
frequencies1 = np.fft.fftfreq(n1, d=1/fs)

positive_freqs1 = frequencies1[:n1//2]
positive_fft1 = np.abs(fft_values1[:n1//2])

theta_demeaned2 = C8_C2_S1_theta - np.mean(C8_C2_S1_theta)
fft_values2 = np.fft.fft(theta_demeaned2)
frequencies2 = np.fft.fftfreq(n2, d=1/fs)

positive_freqs2 = frequencies2[:n2//2]
positive_fft2 = np.abs(fft_values2[:n2//2])

theta_demeaned3 = C8_C3_S1_theta - np.mean(C8_C3_S1_theta)
fft_values3 = np.fft.fft(theta_demeaned3)
frequencies3 = np.fft.fftfreq(n3, d=1/fs)

positive_freqs3 = frequencies3[:n3//2]
positive_fft3 = np.abs(fft_values3[:n3//2])

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")
ax.plot(positive_freqs3, positive_fft3, color="green",label="Exp. 3")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP3/TP_Chaos/Figures/Config8_Cond2and3_spectral_analysis.pdf")


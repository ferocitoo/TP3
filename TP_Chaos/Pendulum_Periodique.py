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




#Import the data
C7_Droite = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config7_Droite.csv", delimiter=';', decimal=',')
C7_Droite_t=C7_Droite.iloc[:,0]
C7_Droite_thetadot=C7_Droite.iloc[:,1]
C7_Droite_theta=C7_Droite.iloc[:,2]

C7_Gauche = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config7_Gauche.csv", delimiter=';', decimal=',')
C7_Gauche_t=C7_Gauche.iloc[:,0]
C7_Gauche_thetadot=C7_Gauche.iloc[:,1]
C7_Gauche_theta=C7_Gauche.iloc[:,2]

C7_GaucheDroite = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config7_GaucheDroite.csv", delimiter=';', decimal=',')
C7_GaucheDroite_t=C7_GaucheDroite.iloc[:,0]
C7_GaucheDroite_thetadot=C7_GaucheDroite.iloc[:,1]
C7_GaucheDroite_theta=C7_GaucheDroite.iloc[:,2]

C8_C1_S1 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config8_Cond1_Simul1.csv", delimiter=';', decimal=',')
C8_C1_S1_t=C8_C1_S1.iloc[:,0]
C8_C1_S1_thetadot=C8_C1_S1.iloc[:,1]
C8_C1_S1_theta=C8_C1_S1.iloc[:,2]

C8_C1_S2 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config8_Cond1_Simul2.csv", delimiter=';', decimal=',')
C8_C1_S2_t=C8_C1_S2.iloc[:,0]
C8_C1_S2_thetadot=C8_C1_S2.iloc[:,1]
C8_C1_S2_theta=C8_C1_S2.iloc[:,2]

C8_C2_S1 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config8_Cond2_Simul1.csv", delimiter=';', decimal=',')
C8_C2_S1_t=C8_C2_S1.iloc[:,0]
C8_C2_S1_thetadot=C8_C2_S1.iloc[:,1]
C8_C2_S1_theta=C8_C2_S1.iloc[:,2]

C8_C3_S1 = pd.read_csv("/workspaces/TP-Chaos/TP_Chaos/Datas/Config8_Cond3_Simul1.csv", delimiter=';', decimal=',')
C8_C3_S1_t=C8_C3_S1.iloc[:,0]
C8_C3_S1_thetadot=C8_C3_S1.iloc[:,1]
C8_C3_S1_theta=C8_C3_S1.iloc[:,2]


#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C7_Droite_t,C7_Droite_theta, label="Right potential well", color="blue")
ax.plot(C7_Gauche_t,C7_Gauche_theta, label="Left potential well", color="red")
ax.plot(C7_GaucheDroite_t,C7_GaucheDroite_theta, label="Both potential wells", color="green")

timerange = [-2,20]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config7_position.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C7_Droite_t,C7_Droite_thetadot, label="Right potential well", color="blue")
ax.plot(C7_Gauche_t,C7_Gauche_thetadot, label="Left potential well", color="red")
ax.plot(C7_GaucheDroite_t,C7_GaucheDroite_thetadot, label="Both potential wells", color="green")

timerange = [-2,20]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config7_angul_speed.pdf")


#phase space
ylabel = r"$\dot{\theta}$" + " [V]"
xlabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C7_Droite_theta,C7_Droite_thetadot, label="Right potential well", color="blue")
ax.plot(C7_Gauche_theta,C7_Gauche_thetadot, label="Left potential well", color="red")
ax.plot(C7_GaucheDroite_theta,C7_GaucheDroite_thetadot, label="Both potential wells", color="green")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config7_phase_space.pdf")


#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
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

norm1 = simps(positive_fft1,x = positive_freqs1)
norm2 = simps(positive_fft2,x = positive_freqs2)
norm3 = simps(positive_fft3,x = positive_freqs3)

positive_fft1 = positive_fft1/norm1
positive_fft2 = positive_fft2/norm2
positive_fft3 = positive_fft3/norm3

max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
max_freq2 = positive_freqs2[np.argmax(positive_fft2)]
max_freq3 = positive_freqs3[np.argmax(positive_fft3)]

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Right potential well")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Left potential well")
ax.plot(positive_freqs3, positive_fft3, color="green",label="Both potential wells")

ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")
ax.axvline(x=max_freq3, color="green", linestyle="--",label=rf"x = {max_freq3*10:.2f} $\times 10^{{{-1}}}$ Hz")

# ax.axvline(x=2*max_freq1, color="black", linestyle="--",label=r"$2 \nu^\star$")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config7_spectral_analysis.pdf")







#Config8Cond1
C8_C1_S1_theta = C8_C1_S1_theta.to_numpy()[200:]
C8_C1_S1_t = C8_C1_S1_t.to_numpy()[200:] - C8_C1_S1_t.to_numpy()[200]
C8_C1_S1_thetadot = C8_C1_S1_thetadot.to_numpy()[200:]

C8_C1_S2_theta = C8_C1_S2_theta.to_numpy()[15:]
C8_C1_S2_t = C8_C1_S2_t.to_numpy()[15:] - C8_C1_S2_t.to_numpy()[15]
C8_C1_S2_thetadot = C8_C1_S2_thetadot.to_numpy()[15:]




#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_t,C8_C1_S1_theta, label="In. Cond. 1", color="blue")
ax.plot(C8_C1_S2_t,C8_C1_S2_theta, label="In. Cond. 2", color="red")

timerange = [-2,35]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond1_position.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_t,C8_C1_S1_thetadot, label="In. Cond. 1", color="blue")
ax.plot(C8_C1_S2_t,C8_C1_S2_thetadot, label="In. Cond. 2", color="red")

timerange = [-2,35]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond1_angul_speed.pdf")


#phase space
ylabel = r"$\dot{\theta}$" + " [V]"
xlabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(C8_C1_S1_theta,C8_C1_S1_thetadot, label="In. Cond. 1", color="blue")
ax.plot(C8_C1_S2_theta,C8_C1_S2_thetadot, label="In. Cond. 2", color="red")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond1_phase_space.pdf")


#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
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

norm1 = simps(positive_fft1,x = positive_freqs1)
norm2 = simps(positive_fft2,x = positive_freqs2)
positive_fft1 = positive_fft1/norm1
positive_fft2 = positive_fft2/norm2

ax.plot(positive_freqs1, positive_fft1, color="blue",label="In. Cond. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="In. Cond. 2")

max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
max_freq2 = positive_freqs2[np.argmax(positive_fft2)]

ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")

# ax.axvline(x=(3)*max_freq1, color="blue", linestyle="--",label=r"$\nu^\star + \frac{2}{3}\nu^\star$")
# ax.axvline(x=(1/3)*max_freq2, color="red", linestyle="--",label=r"$\nu^\star + \frac{2}{3}\nu^\star$")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond1_spectral_analysis.pdf")




theta1 = C8_C1_S1_theta
theta2 = C8_C1_S2_theta[:len(C8_C1_S1_theta)]
t = C8_C1_S1_t

thetadot1 = C8_C1_S1_thetadot
thetadot2 = C8_C1_S2_thetadot[:len(C8_C1_S1_thetadot)]

#Get the excitation frequency of the system 
exc_freq = positive_freqs1[np.argmax(positive_fft1)]


#Calculate the Lyapunov exponent
delta_phase = np.sqrt((theta1-theta2)**2 + ((thetadot1-thetadot2))**2)

#make an exponential fit on the first half of the data
# delta_phase_fit = delta_phase[85:113]
# tfit = t[85:113]

delta_phase_fit = delta_phase[200:]
tfit = t[200:]

# ax.scatter(t[113],delta_phase[113],color="red")
# ax.scatter(t[85],delta_phase[85],color="red")

fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta$ [a.u.]")
ax.plot(t,delta_phase, label="Data", color="blue")
ax.set_yscale("log")



#plot the fit
ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10e3:.1f}\times 10^{{{-3}}})t}}$", color="red", linestyle="--")   

ax.set_xlim()
ax.set_ylim([1e-2,1e1])

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond1_lyapunov.pdf")





# #Config8Cond2and3
# #position 

# xlabel = "Time [s]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# # ax.plot(C8_C1_S1_t,C8_C1_S1_theta, label="Condition initiale 1", color="blue")
# ax.plot(C8_C2_S1_t,C8_C2_S1_theta, label="Condition initiale 2", color="red")
# ax.plot(C8_C3_S1_t,C8_C3_S1_theta, label="Condition initiale 3", color="green")

# timerange = [-2,120]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond2and3_position.pdf")


# #angular speed
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [rad]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# # ax.plot(C8_C1_S1_t,C8_C1_S1_thetadot, label="Condition initiale 1", color="blue")
# ax.plot(C8_C2_S1_t,C8_C2_S1_thetadot, label="Condition initiale 2", color="red")
# ax.plot(C8_C3_S1_t,C8_C3_S1_thetadot, label="Condition initiale 3", color="green")

# timerange = [-2,120]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond2and3_angul_speed.pdf")


# #phase space
# xlabel = r"$\dot{\theta}$" + " [rad]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# # ax.plot(C8_C1_S1_thetadot,C8_C1_S1_theta, label="Condition initiale 1", color="blue")
# ax.plot(C8_C2_S1_thetadot,C8_C2_S1_theta, label="Condition initiale 2", color="red")
# ax.plot(C8_C3_S1_thetadot,C8_C3_S1_theta, label="Condition initiale 3", color="green")

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond2and3_phase_space.pdf")


# #spectral analysis
# xlabel = "Frequency [Hz]"
# ylabel = "Amplitude"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# fs = 1/(C8_C1_S1_t[1]-C8_C1_S1_t[0])

# # n1 = len(C8_C1_S1_theta)
# n2 = len(C8_C2_S1_theta)
# n3 = len(C8_C3_S1_theta)


# #FFT
# # theta_demeaned1 = C8_C1_S1_theta - np.mean(C8_C1_S1_theta)
# # fft_values1 = np.fft.fft(theta_demeaned1)
# # frequencies1 = np.fft.fftfreq(n1, d=1/fs)

# # positive_freqs1 = frequencies1[:n1//2]
# # positive_fft1 = np.abs(fft_values1[:n1//2])

# theta_demeaned2 = C8_C2_S1_theta - np.mean(C8_C2_S1_theta)
# fft_values2 = np.fft.fft(theta_demeaned2)
# frequencies2 = np.fft.fftfreq(n2, d=1/fs)

# positive_freqs2 = frequencies2[:n2//2]
# positive_fft2 = np.abs(fft_values2[:n2//2])

# theta_demeaned3 = C8_C3_S1_theta - np.mean(C8_C3_S1_theta)
# fft_values3 = np.fft.fft(theta_demeaned3)
# frequencies3 = np.fft.fftfreq(n3, d=1/fs)

# positive_freqs3 = frequencies3[:n3//2]
# positive_fft3 = np.abs(fft_values3[:n3//2])

# # ax.plot(positive_freqs1, positive_fft1, color="blue",label="Condition initiale 1")
# ax.plot(positive_freqs2, positive_fft2, color="red",label="Condition initiale 2")
# ax.plot(positive_freqs3, positive_fft3, color="green",label="Condition initiale 3")

# freq_range=[0,1]
# ax.set_xlim(freq_range)
# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config8_Cond2and3_spectral_analysis.pdf")



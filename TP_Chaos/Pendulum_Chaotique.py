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
C1_C0_S1 = pd.read_csv("TP_Chaos/Datas/Config1_Cond0_Simul1.csv", delimiter=';', decimal=',')
C1_C0_S1_t=C1_C0_S1.iloc[:,0]
C1_C0_S1_thetadot=C1_C0_S1.iloc[:,1]
C1_C0_S1_theta=C1_C0_S1.iloc[:,2]

C1_C1_S1 = pd.read_csv("TP_Chaos/Datas/Config1_Cond1_Simul1.csv", delimiter=';', decimal=',')
C1_C1_S1_t=C1_C1_S1.iloc[:,0]
C1_C1_S1_thetadot=C1_C1_S1.iloc[:,1]
C1_C1_S1_theta=C1_C1_S1.iloc[:,2]

C1_C1_S2 = pd.read_csv("TP_Chaos/Datas/Config1_Cond1_Simul2.csv", delimiter=';', decimal=',')
C1_C1_S2_t=C1_C1_S2.iloc[:,0]
C1_C1_S2_theta=C1_C1_S2.iloc[:,1]
C1_C1_S2_thetadot=C1_C1_S2.iloc[:,2]

C1_C1_S3 = pd.read_csv("TP_Chaos/Datas/Config1_Cond1_Simul3.csv", delimiter=';', decimal=',')
C1_C1_S3_t=C1_C1_S3.iloc[:,0]
C1_C1_S3_theta=C1_C1_S3.iloc[:,1]
C1_C1_S3_thetadot=C1_C1_S3.iloc[:,2]

C1_C1_S4 = pd.read_csv("TP_Chaos/Datas/Config1_Cond1_Simul4.csv", delimiter=';', decimal=',')
C1_C1_S4_t=C1_C1_S4.iloc[:,0]
C1_C1_S4_theta=C1_C1_S4.iloc[:,1]
C1_C1_S4_thetadot=C1_C1_S4.iloc[:,2]

C5_C0_S1 = pd.read_csv("TP_Chaos/Datas/Config5_Cond0_Simul1.csv", delimiter=';', decimal=',')
C5_C0_S1_t=C5_C0_S1.iloc[:,0]
C5_C0_S1_theta=C5_C0_S1.iloc[:,1]
C5_C0_S1_thetadot=C5_C0_S1.iloc[:,2]

C5_C0_S2 = pd.read_csv("TP_Chaos/Datas/Config5_Cond0_Simul2.csv", delimiter=';', decimal=',')
C5_C0_S2_t=C5_C0_S2.iloc[:,0]
C5_C0_S2_theta=C5_C0_S2.iloc[:,1]
C5_C0_S2_thetadot=C5_C0_S2.iloc[:,2]

C5_C1_S1 = pd.read_csv("TP_Chaos/Datas/Config5_Cond1_Simul1.csv", delimiter=';', decimal=',')
C5_C1_S1_t=C5_C1_S1.iloc[:,0]
C5_C1_S1_theta=C5_C1_S1.iloc[:,1]   
C5_C1_S1_thetadot=C5_C1_S1.iloc[:,2]

C5_C1_S2 = pd.read_csv("TP_Chaos/Datas/Config5_Cond1_Simul2.csv", delimiter=';', decimal=',')
C5_C1_S2_t=C5_C1_S2.iloc[:,0]
C5_C1_S2_theta=C5_C1_S2.iloc[:,1]
C5_C1_S2_thetadot=C5_C1_S2.iloc[:,2]

C5_C2_S1 = pd.read_csv("TP_Chaos/Datas/Config5_Cond2_Simul1.csv", delimiter=';', decimal=',')
C5_C2_S1_t=C5_C2_S1.iloc[:,0]
C5_C2_S1_theta=C5_C2_S1.iloc[:,1]
C5_C2_S1_thetadot=C5_C2_S1.iloc[:,2]

C5_C2_S2 = pd.read_csv("TP_Chaos/Datas/Config5_Cond2_Simul2.csv", delimiter=';', decimal=',')
C5_C2_S2_t=C5_C2_S2.iloc[:,0]
C5_C2_S2_theta=C5_C2_S2.iloc[:,1]
C5_C2_S2_thetadot=C5_C2_S2.iloc[:,2]

C6_C1_S1 = pd.read_csv("TP_Chaos/Datas/Config6_Cond1_Simul1.csv", delimiter=';', decimal=',')
C6_C1_S1_t=C6_C1_S1.iloc[:,0]
C6_C1_S1_theta=C6_C1_S1.iloc[:,1]
C6_C1_S1_thetadot=C6_C1_S1.iloc[:,2]

C6_C1_S2 = pd.read_csv("TP_Chaos/Datas/Config6_Cond1_Simul2.csv", delimiter=';', decimal=',')
C6_C1_S2_t=C6_C1_S2.iloc[:,0]
C6_C1_S2_theta=C6_C1_S2.iloc[:,1]
C6_C1_S2_thetadot=C6_C1_S2.iloc[:,2]

C7_Droite = pd.read_csv("TP_Chaos/Datas/Config7_Droite.csv", delimiter=';', decimal=',')
C7_Droite_t=C7_Droite.iloc[:,0]
C7_Droite_theta=C7_Droite.iloc[:,1]
C7_Droite_thetadot=C7_Droite.iloc[:,2]

C7_Gauche = pd.read_csv("TP_Chaos/Datas/Config7_Gauche.csv", delimiter=';', decimal=',')
C7_Gauche_t=C7_Gauche.iloc[:,0]
C7_Gauche_theta=C7_Gauche.iloc[:,1]
C7_Gauche_thetadot=C7_Gauche.iloc[:,2]

C7_GaucheDroite = pd.read_csv("TP_Chaos/Datas/Config7_GaucheDroite.csv", delimiter=';', decimal=',')
C7_GaucheDroite_t=C7_GaucheDroite.iloc[:,0]
C7_GaucheDroite_theta=C7_GaucheDroite.iloc[:,1]
C7_GaucheDroite_thetadot=C7_GaucheDroite.iloc[:,2]

CJOLI = pd.read_csv("TP_Chaos/Datas/Config5_JOLI.csv", delimiter=';', decimal=',')
CJOLI_t=CJOLI.iloc[:,0]
CJOLI_theta=CJOLI.iloc[:,1]
CJOLI_thetadot=CJOLI.iloc[:,2]








# #----------Configuration 5, Condition 2, Simulation 1 and 2----------#

# #--Plot--#
# t1 = C5_C2_S1_t
# theta1 = C5_C2_S1_theta
# thetadot1 = C5_C2_S1_thetadot

# t2 = C5_C2_S2_t
# theta2 = C5_C2_S2_theta
# thetadot2 = C5_C2_S2_thetadot


# #position
# xlabel = "Time [s]"
# ylabel = r"\theta" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,theta1, label="Exp. 1", color="blue")
# ax.plot(t2,theta2, label="Exp. 2", color="red")

# timerange = [-2,30]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_angle.pdf")


# #angular speed
# xlabel = "Time [s]"
# ylabel = r"\dot{\theta}" + " [rad]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
# ax.plot(t2,thetadot2, label="Exp. 2", color="red")

# timerange = [-2,30]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_speed.pdf")



# #phase space
# xlabel = r"\dot{\theta}" + " [V]"
# ylabel = r"\theta" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
# ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_phase_space.pdf")




# #spectral analysis
# xlabel = r"$\nu$ [Hz]"
# ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# fs1 = 1/(t1[1]-t1[0])
# fs2 = 1/(t2[1]-t2[0])

# n1 = len(theta1)
# n2 = len(theta2)

# #FFT 
# theta_demeaned1 = theta1 - np.mean(theta1)
# theta_demeaned1 = np.pad(theta_demeaned1, (0, n1), 'constant')
# n1 = len(theta_demeaned1)
# fft_values1 = np.fft.fft(theta_demeaned1)
# frequencies1 = np.fft.fftfreq(n1, d=1/fs1)
# positive_freqs1 = frequencies1[:n1//2]
# positive_fft1 = np.abs(fft_values1[:n1//2])

# theta_demeaned2 = theta2 - np.mean(theta2)
# theta_demeaned2 = np.pad(theta_demeaned2, (0, n2), 'constant')
# n2 = len(theta_demeaned2)
# fft_values2 = np.fft.fft(theta_demeaned2)
# frequencies2 = np.fft.fftfreq(n2, d=1/fs2)
# positive_freqs2 = frequencies2[:n2//2]
# positive_fft2 = np.abs(fft_values2[:n2//2])

# norm1 = simps(positive_fft1,x = positive_freqs1)
# norm2 = simps(positive_fft2,x = positive_freqs2)
# positive_fft1 = positive_fft1/norm1
# positive_fft2 = positive_fft2/norm2

# ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
# ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

# max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
# max_freq2 = positive_freqs2[np.argmax(positive_fft2)]

# # ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
# # ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")

# print("Max frequency, C5_C2_S1/S2: ",max_freq1,max_freq2)


# freq_range=[0,1]
# ax.set_xlim(freq_range)
# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_spectral_analysis.pdf")



# theta2 = theta2[:len(theta1)]
# t=t1


# thetadot2 = thetadot2[:len(thetadot1)]

# #Get the excitation frequency of the system 
# exc_freq = positive_freqs1[np.argmax(positive_fft1)]

# #Calculate the Lyapunov exponent
# delta_phase = np.sqrt((theta1-theta2)**2 + ((thetadot1-thetadot2)/2*np.pi*exc_freq)**2)

# #make an exponential fit on the first half of the data
# # delta_phase_fit = delta_phase[85:113]
# # tfit = t[85:113]

# delta_phase_fit = delta_phase[150:540]
# tfit = t1[150:540]



# fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

# ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta$ [a.u.]")
# ax.plot(t,delta_phase, label="Data", color="blue")
# ax.set_yscale("log")

# # ax.scatter(t[540],delta_phase[540],color="red",s=30)
# # ax.scatter(t[150],delta_phase[150],color="red",s=30)


# #plot the fit
# ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10:.2f} \times 10^{{{-1}}})t}} $", color="red", linestyle="--")   

# ax.set_xlim([0,50])
# ax.set_ylim([1e-4,1e1])

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config5_Cond2_lyapunov.pdf")



# #----------Configuration 6, Condition 1, Simulation 1 and 2----------#

# #--Plot--#
# t1 = C6_C1_S1_t
# theta1 = C6_C1_S1_theta
# thetadot1 = C6_C1_S1_thetadot

# t2 = C6_C1_S2_t
# theta2 = C6_C1_S2_theta
# thetadot2 = C6_C1_S2_thetadot


# #position
# xlabel = "Time [s]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,theta1, label="Exp. 1", color="blue")
# ax.plot(t2,theta2, label="Exp. 2", color="red")

# timerange = [-2,30]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_angle.pdf")


# #angular speed
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
# ax.plot(t2,thetadot2, label="Exp. 2", color="red")

# timerange = [-2,30]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_speed.pdf")


# #phase space
# xlabel = r"$\dot{\theta}$" + " [V]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
# ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_phase_space.pdf")


# #spectral analysis
# xlabel = r"$\nu$ [Hz]"
# ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# fs1 = 1/(t1[1]-t1[0])
# fs2 = 1/(t2[1]-t2[0])

# n1 = len(theta1)
# n2 = len(theta2)

# #FFT
# theta_demeaned1 = theta1 - np.mean(theta1)
# theta_demeaned1 = np.pad(theta_demeaned1, (0, n1), 'constant')
# n1 = len(theta_demeaned1)
# fft_values1 = np.fft.fft(theta_demeaned1)
# frequencies1 = np.fft.fftfreq(n1, d=1/fs1)

# positive_freqs1 = frequencies1[:n1//2]
# positive_fft1 = np.abs(fft_values1[:n1//2])

# theta_demeaned2 = theta2 - np.mean(theta2)
# theta_demeaned2 = np.pad(theta_demeaned2, (0, n2), 'constant')
# n2 = len(theta_demeaned2)
# fft_values2 = np.fft.fft(theta_demeaned2)
# frequencies2 = np.fft.fftfreq(n2, d=1/fs2)

# positive_freqs2 = frequencies2[:n2//2]
# positive_fft2 = np.abs(fft_values2[:n2//2])

# norm1 = simps(positive_fft1,x=positive_freqs1)
# norm2 = simps(positive_fft2,x=positive_freqs2)
# positive_fft1 = positive_fft1/norm1
# positive_fft2 = positive_fft2/norm2

# ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
# ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

# max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
# max_freq2 = positive_freqs2[np.argmax(positive_fft2)]

# # ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
# # ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")

# print("Max frequency, C6_C1_S1/S2: ",max_freq1,max_freq2)


# freq_range=[0,1]
# ax.set_xlim(freq_range)
# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_spectral_analysis.pdf")



# #LYAPUNOV EXPONENT
# theta2 = theta2[:len(theta1)]
# t=t1


# thetadot2 = thetadot2[:len(thetadot1)]

# #Get the excitation frequency of the system 
# exc_freq = positive_freqs1[np.argmax(positive_fft1)]

# #Calculate the Lyapunov exponent
# delta_phase = np.sqrt((theta1-theta2)**2 + ((thetadot1-thetadot2)/2*np.pi*exc_freq)**2)

# #make an exponential fit on the first half of the data
# # delta_phase_fit = delta_phase[85:113]
# # tfit = t[85:113]

# delta_phase_fit = delta_phase[75:575]
# tfit = t1[75:575]



# fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

# ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta$ [a.u.]")
# ax.plot(t,delta_phase, label="Data", color="blue")
# ax.set_yscale("log")

# # ax.scatter(t[575],delta_phase[575],color="red",s=30)
# # ax.scatter(t[75],delta_phase[75],color="red",s=30)


# #plot the fit
# ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10:.2f} \times 10^{{{-1}}})t}}  $", color="red", linestyle="--")   

# ax.set_xlim([0,50])
# ax.set_ylim([1e-4,1e1])

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config6_Cond1_lyapunov.pdf")



#----------Configuration 1 Condition 1, Simulation 1,2,3 and 4----------#

#--Plot--#
t1 = C1_C1_S1_t
theta1 = C1_C1_S1_theta
thetadot1 = C1_C1_S1_thetadot

t2 = C1_C1_S2_t
theta2 = C1_C1_S2_theta
thetadot2 = C1_C1_S2_thetadot

t3 = C1_C1_S3_t
theta3 = C1_C1_S3_theta
thetadot3 = C1_C1_S3_thetadot

t4 = C1_C1_S4_t
theta4 = C1_C1_S4_theta
thetadot4 = C1_C1_S4_thetadot

#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 2", color="red")
ax.plot(t3,theta3, label="Exp. 3", color="green")
ax.plot(t4,theta4, label="Exp. 4", color="orange")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_angle.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 2", color="red")
ax.plot(t3,thetadot3, label="Exp. 3", color="green")
ax.plot(t4,thetadot4, label="Exp. 4", color="orange")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_speed.pdf")



#phase space
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 2")
ax.plot(thetadot3,theta3,color="green",label="Exp. 3")
ax.plot(thetadot4,theta4,color="orange",label="Exp. 4")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_phase_space.pdf")


#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs1 = 1/(t1[1]-t1[0])
fs2 = 1/(t2[1]-t2[0])
fs3 = 1/(t3[1]-t3[0])
fs4 = 1/(t4[1]-t4[0])


n1 = len(theta1)
n2 = len(theta2)
n3 = len(theta3)
n4 = len(theta4)


#FFT
theta_demeaned1 = theta1 - np.mean(theta1)
theta_demeaned1 = np.pad(theta_demeaned1, (0, n1), 'constant')
n1 = len(theta_demeaned1)
fft_values1 = np.fft.fft(theta_demeaned1)
frequencies1 = np.fft.fftfreq(n1, d=1/fs1)
positive_freqs1 = frequencies1[:n1//2]
positive_fft1 = np.abs(fft_values1[:n1//2])

theta_demeaned2 = theta2 - np.mean(theta2)
theta_demeaned2 = np.pad(theta_demeaned2, (0, n2), 'constant')
n2 = len(theta_demeaned2)
fft_values2 = np.fft.fft(theta_demeaned2)
frequencies2 = np.fft.fftfreq(n2, d=1/fs2)
positive_freqs2 = frequencies2[:n2//2]
positive_fft2 = np.abs(fft_values2[:n2//2])

theta_demeaned3 = theta3 - np.mean(theta3)
theta_demeaned3 = np.pad(theta_demeaned3, (0, n3), 'constant')
n3 = len(theta_demeaned3)
fft_values3 = np.fft.fft(theta_demeaned3)
frequencies3 = np.fft.fftfreq(n3, d=1/fs3)
positive_freqs3 = frequencies3[:n3//2]
positive_fft3 = np.abs(fft_values3[:n3//2])

theta_demeaned4 = theta4 - np.mean(theta4)
theta_demeaned4 = np.pad(theta_demeaned4, (0, n4), 'constant')
n4 = len(theta_demeaned4)
fft_values4 = np.fft.fft(theta_demeaned4)
frequencies4 = np.fft.fftfreq(n4, d=1/fs4)
positive_freqs4 = frequencies4[:n4//2]
positive_fft4 = np.abs(fft_values4[:n4//2])



norm1 = simps(positive_fft1,x = positive_freqs1)
norm2 = simps(positive_fft2,x = positive_freqs2)
norm3 = simps(positive_fft3,x = positive_freqs3)
norm4 = simps(positive_fft4,x = positive_freqs4)

positive_fft1 = positive_fft1/norm1
positive_fft2 = positive_fft2/norm2
positive_fft3 = positive_fft3/norm3
positive_fft4 = positive_fft4/norm4

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")
ax.plot(positive_freqs3, positive_fft3, color="green",label="Exp. 3")
ax.plot(positive_freqs4, positive_fft4, color="orange",label="Exp. 4")

max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
max_freq2 = positive_freqs2[np.argmax(positive_fft2)]
max_freq3 = positive_freqs3[np.argsort(positive_fft3)[-2]]
max_freq4 = positive_freqs4[np.argmax(positive_fft4)]

# ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
# ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")
# ax.axvline(x=max_freq3, color="green", linestyle="--",label=rf"x = {max_freq3*10:.2f} $\times 10^{{{-1}}}$ Hz")
# ax.axvline(x=max_freq4, color="orange", linestyle="--",label=rf"x = {max_freq4*10:.2f} $\times 10^{{{-1}}}$ Hz")

ax.axvline(x=(5/3)*max_freq3, color="black", linestyle="--",label = r"$\nu^\star + \frac{2}{3}\nu^\star$")
ax.axvline(x=(1/3)*max_freq3, color="black", linestyle="-.",label = r"$\nu^\star - \frac{2}{3}\nu^\star$")

print("Max frequency, C1_C1_S1/S2/S3/S4: ",max_freq1,max_freq2,max_freq3,max_freq4)

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_spectral_analysis.pdf")




#LYAPUNOV EXPONENT

n1 = len(theta1)
n2 = len(theta2)
n3 = len(theta3)
n4 = len(theta4)

theta1 = theta1[:min(n1,n2,n3,n4)]
theta2 = theta2[:min(n1,n2,n3,n4)]
theta3 = theta3[:min(n1,n2,n3,n4)]
theta4 = theta4[:min(n1,n2,n3,n4)]

thetadot1 = thetadot1[:min(n1,n2,n3,n4)]
thetadot2 = thetadot2[:min(n1,n2,n3,n4)]
thetadot3 = thetadot3[:min(n1,n2,n3,n4)]
thetadot4 = thetadot4[:min(n1,n2,n3,n4)]

t=t1[:min(n1,n2,n3,n4)]

exc_freq = positive_freqs1[np.argmax(positive_fft1)]

#Calculate the Lyapunov exponent
delta_phase12 = np.sqrt((theta1-theta2)**2 + ((thetadot1-thetadot2)/2*np.pi*exc_freq)**2)
delta_phase13 = np.sqrt((theta1-theta3)**2 + ((thetadot1-thetadot3)/2*np.pi*exc_freq)**2)
delta_phase14 = np.sqrt((theta1-theta4)**2 + ((thetadot1-thetadot4)/2*np.pi*exc_freq)**2)
delta_phase23 = np.sqrt((theta2-theta3)**2 + ((thetadot2-thetadot3)/2*np.pi*exc_freq)**2)
delta_phase24 = np.sqrt((theta2-theta4)**2 + ((thetadot2-thetadot4)/2*np.pi*exc_freq)**2)
delta_phase34 = np.sqrt((theta3-theta4)**2 + ((thetadot3-thetadot4)/2*np.pi*exc_freq)**2)

delta = (delta_phase12 + delta_phase13 + delta_phase14 + delta_phase23 + delta_phase24 + delta_phase34)/6

#make an exponential fit on the first half of the data
delta_phase_fit = delta_phase34[15:95]
tfit = t[15:95]




fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta_{34}$ [a.u.]")
ax.plot(t,delta_phase34, label="Data", color="blue")
ax.set_yscale("log")

# ax.scatter(t[95],delta_phase34[95],color="red",s=30)
# ax.scatter(t[15],delta_phase34[15],color="red",s=30)

#plot the fit
ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10:.2f} \times 10^{{{-1}}})t}} $", color="red", linestyle="--")

ax.set_xlim([0,50])
ax.set_ylim([1e-4,1e1])

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config1_Cond1_lyapunov34.pdf")




#make an exponential fit on the first half of the data
delta_phase_fit = delta_phase13[25:77]
tfit = t[25:77]


fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta_{12}$ [a.u.]")
ax.plot(t,delta_phase13, label="Data", color="blue")
ax.set_yscale("log")

# ax.scatter(t[77],delta_phase13[77],color="red",s=30)
# ax.scatter(t[25],delta_phase13[25],color="red",s=30)


#plot the fit
ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10:.2f} \times 10^{{{-1}}})t}} $", color="red", linestyle="--")

ax.set_xlim([0,50])
ax.set_ylim([1e-4,1e1])

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config1_Cond1_lyapunov12.pdf")






# #----------Configuration 5 Condition 0, Simulation 1 and 2----------#

# #--Plot--#
# t1 = C5_C0_S1_t
# theta1 = C5_C0_S1_theta
# thetadot1 = C5_C0_S1_thetadot

# t2 = C5_C0_S2_t
# theta2 = C5_C0_S2_theta
# thetadot2 = C5_C0_S2_thetadot

# #position
# xlabel = "Time [s]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,theta1, label="Exp. 1", color="blue")
# ax.plot(t2,theta2, label="Exp. 2", color="red")

# timerange = [-2,30]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_angle.pdf")


# #angular speed
# xlabel = "Time [s]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
# ax.plot(t2,thetadot2, label="Exp. 2", color="red")

# timerange = [-2,30]
# ax.set_xlim(timerange)

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_speed.pdf")



# #phase space
# xlabel = r"$\dot{\theta}$" + " [V]"
# ylabel = r"$\theta$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
# ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_phase_space.pdf")


# #spectral analysis
# xlabel = r"$\nu$ [Hz]"
# ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# fs1 = 1/(t1[1]-t1[0])
# fs2 = 1/(t2[1]-t2[0])

# n1 = len(theta1)
# n2 = len(theta2)


# #FFT
# theta_demeaned1 = theta1 - np.mean(theta1)
# theta_demeaned1 = np.pad(theta_demeaned1, (0, n1), 'constant')
# n1 = len(theta_demeaned1)
# fft_values1 = np.fft.fft(theta_demeaned1)
# frequencies1 = np.fft.fftfreq(n1, d=1/fs1)
# positive_freqs1 = frequencies1[:n1//2]
# positive_fft1 = np.abs(fft_values1[:n1//2])

# theta_demeaned2 = theta2 - np.mean(theta2)
# theta_demeaned2 = np.pad(theta_demeaned2, (0, n2), 'constant')
# n2 = len(theta_demeaned2)
# fft_values2 = np.fft.fft(theta_demeaned2)
# frequencies2 = np.fft.fftfreq(n2, d=1/fs2)
# positive_freqs2 = frequencies2[:n2//2]
# positive_fft2 = np.abs(fft_values2[:n2//2])

# norm1 = simps(positive_fft1,x = positive_freqs1)
# norm2 = simps(positive_fft2,x = positive_freqs2)
# positive_fft1 = positive_fft1/norm1
# positive_fft2 = positive_fft2/norm2

# ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
# ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

# max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
# max_freq2 = positive_freqs2[np.argmax(positive_fft2)]

# # ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
# # ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")

# print("Max frequency, C5_C0_S1/S2: ",max_freq1,max_freq2)

# freq_range=[0,1]
# ax.set_xlim(freq_range)
# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_spectral_analysis.pdf")





# #LYAPUNOV EXPONENT
# theta2 = theta2[:len(theta1)]
# t=t1


# thetadot2 = thetadot2[:len(thetadot1)]

# #Get the excitation frequency of the system 
# exc_freq = positive_freqs1[np.argmax(positive_fft1)]

# #Calculate the Lyapunov exponent
# delta_phase = np.sqrt((theta1-theta2)**2 + ((thetadot1-thetadot2)/2*np.pi*exc_freq)**2)

# #make an exponential fit on the first half of the data
# # delta_phase_fit = delta_phase[85:113]
# # tfit = t[85:113]

# delta_phase_fit = delta_phase[25:100]
# tfit = t1[25:100]



# fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

# ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta$ [a.u.]")
# ax.plot(t,delta_phase, label="Data", color="blue")
# ax.set_yscale("log")

# # ax.scatter(t[100],delta_phase[100],color="red",s=30)
# # ax.scatter(t[25],delta_phase[25],color="red",s=30)


# #plot the fit
# ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10:.2f} \times 10^{{{-1}}})t}} $", color="red", linestyle="--")   

# ax.set_xlim([0,50])
# ax.set_ylim([1e-4,1e1])

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config5_Cond0_lyapunov.pdf")


#----------Configuration 5 Condition 1, Simulation 1 and 2----------#

#--Plot--#
t1 = C5_C1_S1_t
theta1 = C5_C1_S1_theta
thetadot1 = C5_C1_S1_thetadot

t2 = C5_C1_S2_t
theta2 = C5_C1_S2_theta
thetadot2 = C5_C1_S2_thetadot

#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond1_Simul1_2_angle.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond1_Simul1_2_speed.pdf")


#phase space
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond1_Simul1_2_phase_space.pdf")


#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs1 = 1/(t1[1]-t1[0])
fs2 = 1/(t2[1]-t2[0])

n1 = len(theta1)
n2 = len(theta2)

#FFT
theta_demeaned1 = theta1 - np.mean(theta1)
theta_demeaned1 = np.pad(theta_demeaned1, (0, n1), 'constant')
n1 = len(theta_demeaned1)
fft_values1 = np.fft.fft(theta_demeaned1)
frequencies1 = np.fft.fftfreq(n1, d=1/fs1)
positive_freqs1 = frequencies1[:n1//2]
positive_fft1 = np.abs(fft_values1[:n1//2])

theta_demeaned2 = theta2 - np.mean(theta2)
theta_demeaned2 = np.pad(theta_demeaned2, (0, n2), 'constant')
n2 = len(theta_demeaned2)
fft_values2 = np.fft.fft(theta_demeaned2)
frequencies2 = np.fft.fftfreq(n2, d=1/fs2)
positive_freqs2 = frequencies2[:n2//2]
positive_fft2 = np.abs(fft_values2[:n2//2])

norm1 = simps(positive_fft1,x=positive_freqs1)
norm2 = simps(positive_fft2,x=positive_freqs2)
positive_fft1 = positive_fft1/norm1
positive_fft2 = positive_fft2/norm2

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

max_freq1 = positive_freqs1[np.argmax(positive_fft1)]
max_freq2 = positive_freqs2[np.argmax(positive_fft2)]

# ax.axvline(x=max_freq1, color="blue", linestyle="--",label=rf"x = {max_freq1*10:.2f} $\times 10^{{{-1}}}$ Hz")
# ax.axvline(x=max_freq2, color="red", linestyle="--",label=rf"x = {max_freq2*10:.2f} $\times 10^{{{-1}}}$ Hz")

ax.axvline(x=(5/3)*max_freq1, color="black", linestyle="--",label = r"$\nu^\star + \frac{2}{3}\nu^\star$")
ax.axvline(x=(1/3)*max_freq1, color="black", linestyle="-.",label = r"$\nu^\star - \frac{2}{3}\nu^\star$")

print("Max frequency, C5_C1_S1/S2: ",max_freq1,max_freq2)

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond1_Simul1_2_spectral_analysis.pdf")


#LYAPUNOV EXPONENT
theta2 = theta2[:min(n1,n2)]
theta1 = theta1[:min(n1,n2)]
t=t2[:min(n1,n2)]
thetadot2 = thetadot2[:min(n1,n2)]
thetadot2 = thetadot2[:min(n1,n2)]


#Get the excitation frequency of the system
exc_freq = positive_freqs1[np.argmax(positive_fft1)]

#Calculate the Lyapunov exponent
delta_phase = np.sqrt((theta1-theta2)**2 + ((thetadot1-thetadot2)/2*np.pi*exc_freq)**2)

#make an exponential fit on the first half of the data
# delta_phase_fit = delta_phase[85:113]
# tfit = t[85:113]

delta_phase_fit = delta_phase[25:100]
tfit = t1[25:100]


fit = np.polyfit(tfit,np.log(delta_phase_fit),1)

ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Time [s]", ylabel=r"$\delta$ [a.u.]")
ax.plot(t,delta_phase, label="Data", color="blue")
ax.set_yscale("log")

# ax.scatter(t[100],delta_phase[100],color="red",s=30)
# ax.scatter(t[25],delta_phase[25],color="red",s=30)

#plot the fit
ax.plot(t,np.exp(fit[1])*np.exp(fit[0]*t), label=rf"$\delta(t) \propto e^{{({fit[0]*10:.2f} \times 10^{{{-1}}})t}} $", color="red", linestyle="--")

ax.set_xlim([0,50])
ax.set_ylim([1e-4,1e1])

u.set_legend_properties(ax,fontsize=18)
fig.savefig("/workspaces/TP-Chaos/TP_Chaos/Figures/Config5_Cond1_lyapunov.pdf")







#----------Configuration JOLI----------#
#--Plot--#
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
#phase space
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(CJOLI_thetadot,CJOLI_theta,color="black",linestyle=':', linewidth=0.5)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_JOLI_phase_space.pdf")


#spectral analysis
xlabel = r"$\nu$ [Hz]"
ylabel = r"$\mathcal{F}(\theta - \bar{\theta})(\nu)$" + "[a.u]" 
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

fs = 1/(CJOLI_t[1]-CJOLI_t[0])

n = len(CJOLI_theta)

#FFT
theta_demeaned = CJOLI_theta - np.mean(CJOLI_theta)
theta_demeaned = np.pad(theta_demeaned, (0, n), 'constant')
n = len(theta_demeaned)
fft_values = np.fft.fft(theta_demeaned)
frequencies = np.fft.fftfreq(n, d=1/fs)
positive_freqs = frequencies[:n//2]
positive_fft = np.abs(fft_values[:n//2])

ax.plot(positive_freqs, positive_fft, color="black")

max_freq = positive_freqs[np.argmax(positive_fft)]

# ax.axvline(x=max_freq, color="black", linestyle="--",label=rf"x = {max_freq*10:.2f} $\times 10^{{{-1}}}$ Hz")

print("Max frequency, CJoli: ",max_freq)

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_JOLI_spectral_analysis.pdf")


# max_freq = positive_freqs[np.argmax(positive_fft)]

# excitation_freq = max_freq


# #make poincaré section at each t = 2*pi/excitation_freq
# t = CJOLI_t
# theta = CJOLI_theta.to_numpy()
# thetadot = CJOLI_thetadot.to_numpy()

# dt = (t[1]-t[0])

# T = 1/excitation_freq



# poincare_theta = []
# poincare_thetadot = []

# n = 1

# # Sample data at multiples of T and T/2
# for i in range(len(t)):
#     for k in range(n):  # Loop over 8 points in each period
#         if t[i] % T < dt or (t[i] + k*T/n) % T < dt:
#             poincare_theta.append(theta[i])
#             poincare_thetadot.append(thetadot[i])
        
        
# #count true indices
# # print("Number of points in the Poincaré section: ",np.sum(indices))

# #print total time
# print("Total time: ",t.to_numpy()[-1])

# #position
# xlabel = r"$\theta$" + " [V]"
# ylabel = r"$\dot{\theta}$" + " [V]"
# ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)


# ax.scatter(poincare_thetadot,poincare_theta,color="black",s=1.5,marker='x')

# u.set_legend_properties(ax,fontsize=18)
# fig.savefig("TP_Chaos/Figures/Config5_JOLI_poincare_section.pdf")


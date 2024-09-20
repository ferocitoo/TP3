import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os

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
C1_C0_S1_theta=C1_C0_S1.iloc[:,1]
C1_C0_S1_thetadot=C1_C0_S1.iloc[:,2]

C1_C1_S1 = pd.read_csv("TP_Chaos/Datas/Config1_Cond1_Simul1.csv", delimiter=';', decimal=',')
C1_C1_S1_t=C1_C1_S1.iloc[:,0]
C1_C1_S1_theta=C1_C1_S1.iloc[:,1]
C1_C1_S1_thetadot=C1_C1_S1.iloc[:,2]

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








#----------Configuration 5, Condition 2, Simulation 1 and 2----------#

#--Plot--#
t1 = C5_C2_S1_t
theta1 = C5_C2_S1_theta
thetadot1 = C5_C2_S1_thetadot

t2 = C5_C2_S2_t
theta2 = C5_C2_S2_theta
thetadot2 = C5_C2_S2_thetadot


#position
xlabel = "Time [s]"
ylabel = r"\theta" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_angle.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"\dot{\theta}" + " [rad]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_speed.pdf")



#phase space
xlabel = r"\dot{\theta}" + " [rad]"
ylabel = r"\theta" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_phase_space.pdf")




#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
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

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")


freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond2_Simul1_2_spectral_analysis.pdf")





#----------Configuration 6, Condition 1, Simulation 1 and 2----------#

#--Plot--#
t1 = C6_C1_S1_t
theta1 = C6_C1_S1_theta
thetadot1 = C6_C1_S1_thetadot

t2 = C6_C1_S2_t
theta2 = C6_C1_S2_theta
thetadot2 = C6_C1_S2_thetadot


#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_angle.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_speed.pdf")


#phase space
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_phase_space.pdf")


#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
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

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config6_Cond1_Simul1_2_spectral_analysis.pdf")




#----------Configuration 1 Condition 1, Simulation 1,2,3 and 4----------#

#--Plot--#
# t1 = C1_C1_S1_t
# theta1 = C1_C1_S1_theta
# thetadot1 = C1_C1_S1_thetadot

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

# ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 1", color="red")
ax.plot(t3,theta3, label="Exp. 2", color="green")
ax.plot(t4,theta4, label="Exp. 3", color="orange")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_angle.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 1", color="red")
ax.plot(t3,thetadot3, label="Exp. 2", color="green")
ax.plot(t4,thetadot4, label="Exp. 3", color="orange")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_speed.pdf")



#phase space
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

# ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 1")
ax.plot(thetadot3,theta3,color="green",label="Exp. 2")
ax.plot(thetadot4,theta4,color="orange",label="Exp. 3")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_phase_space.pdf")


#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
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

# ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 1")
ax.plot(positive_freqs3, positive_fft3, color="green",label="Exp. 2")
ax.plot(positive_freqs4, positive_fft4, color="orange",label="Exp. 3")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config1_Cond1_Simul1_2_3_4_spectral_analysis.pdf")



#----------Configuration 5 Condition 0, Simulation 1 and 2----------#

#--Plot--#
t1 = C5_C0_S1_t
theta1 = C5_C0_S1_theta
thetadot1 = C5_C0_S1_thetadot

t2 = C5_C0_S2_t
theta2 = C5_C0_S2_theta
thetadot2 = C5_C0_S2_thetadot

#position
xlabel = "Time [s]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_angle.pdf")


#angular speed
xlabel = "Time [s]"
ylabel = r"$\dot{\theta}$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 2", color="red")

timerange = [-2,30]
ax.set_xlim(timerange)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_speed.pdf")



#phase space
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_phase_space.pdf")


#spectral analysis
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
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

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond0_Simul1_2_spectral_analysis.pdf")


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
xlabel = "Frequency [Hz]"
ylabel = "Amplitude"
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

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")

freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_Cond1_Simul1_2_spectral_analysis.pdf")



#----------Configuration JOLI----------#
#--Plot--#
xlabel = r"$\dot{\theta}$" + " [V]"
ylabel = r"$\theta$" + " [V]"
#phase space
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(CJOLI_thetadot,CJOLI_theta,color="black",linestyle=':', linewidth=0.5)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/Config5_JOLI_phase_space.pdf")

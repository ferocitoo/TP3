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







#----------Configuration x, Condition x, Simulation x----------#

#--Plot--#
xlabel = "Temps (s)"
ylabel = "Position (rad)"

# t=C7_Droite_t
# theta = C7_Droite_theta
# thetadot = C7_Droite_thetadot

t1 = C5_C2_S1_t
theta1 = C5_C2_S1_theta
thetadot1 = C5_C2_S1_thetadot

t2 = C5_C2_S2_t
theta2 = C5_C2_S2_theta
thetadot2 = C5_C2_S2_thetadot

#position
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,theta1, label="Exp. 1", color="blue")
ax.plot(t2,theta2, label="Exp. 2", color="red")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/ConfigX_CondX_SimulX_angle.pdf")


#angular speed
ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

ax.plot(t1,thetadot1, label="Exp. 1", color="blue")
ax.plot(t2,thetadot2, label="Exp. 2", color="red")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/ConfigX_CondX_SimulX_speed.pdf")



#phase space
ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Position (rad)", ylabel="Vitesse angulaire (rad/s)")

ax.plot(thetadot1,theta1,color="blue",label="Exp. 1")
ax.plot(thetadot2,theta2,color="red",label="Exp. 2")

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/ConfigX_CondX_SimulX_phase_space.pdf")





#spectral analysis
ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Fréquence (Hz)", ylabel="Amplitude")

fs = 1/(t1[1]-t1[0])

n1 = len(theta1)
n2 = len(theta2)

#FFT 
theta_demeaned1 = theta1 - np.mean(theta1)
fft_values1 = np.fft.fft(theta_demeaned1)
frequencies1 = np.fft.fftfreq(n1, d=1/fs)
positive_freqs1 = frequencies1[:n1//2]
positive_fft1 = np.abs(fft_values1[:n1//2])

theta_demeaned2 = theta2 - np.mean(theta2)
fft_values2 = np.fft.fft(theta_demeaned2)
frequencies2 = np.fft.fftfreq(n2, d=1/fs)
positive_freqs2 = frequencies2[:n2//2]
positive_fft2 = np.abs(fft_values2[:n2//2])

ax.plot(positive_freqs1, positive_fft1, color="blue",label="Exp. 1")
ax.plot(positive_freqs2, positive_fft2, color="red",label="Exp. 2")


freq_range=[0,1]
ax.set_xlim(freq_range)
u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Chaos/Figures/ConfigX_CondX_SimulX_spectral_analysis.pdf")












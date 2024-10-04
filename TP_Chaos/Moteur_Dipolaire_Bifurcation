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



#erase the rows which have the first charachter being a "T" in the csv file
def erase_rows(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path, "w") as f:
        for line in lines:
            if line[0] != "T":
                f.write(line)
                
erase_rows("TP_Chaos/Datas_Moteur/bifurc_18.csv")


def count_time(time_array,N) : 
    full_cycles = 0
    for i in range(len(time_array) - 1):
        if time_array[i] == 32 and time_array[i + 1] == 0:
            full_cycles += 1

    # Calculate the total time from full cycles
    total_time = full_cycles * 32.0
    if time_array[-1] < 32:
        total_time += time_array[-1]
    return np.linspace(0,total_time,N)

#Import the data
#bifurc_3
Bif_3 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_3.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_3_t=Bif_3.iloc[:,0]
Bif_3_theta=Bif_3.iloc[:,1]
Bif_3_thetadot=Bif_3.iloc[:,2]

#to numpy 
Bif_3_theta = Bif_3_theta.to_numpy()
Bif_3_thetadot = Bif_3_thetadot.to_numpy()

#count time
time_array = Bif_3_t.to_numpy()
Bif_3_t = count_time(time_array,len(Bif_3_theta))



#bifurc_4 
Bif_4 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_4.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_4_t=Bif_4.iloc[:,0]
Bif_4_theta=Bif_4.iloc[:,1]
Bif_4_thetadot=Bif_4.iloc[:,2]

#to numpy
Bif_4_theta = Bif_4_theta.to_numpy()
Bif_4_thetadot = Bif_4_thetadot.to_numpy()

#count time
time_array = Bif_4_t.to_numpy() 
Bif_4_t = count_time(time_array,len(Bif_4_theta))


#bifurc_2
Bif_2 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_2.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_2_t=Bif_2.iloc[:,0]
Bif_2_thetadot=Bif_2.iloc[:,1]
Bif_2_theta=Bif_2.iloc[:,2]

#to numpy
Bif_2_theta = Bif_2_theta.to_numpy()
Bif_2_thetadot = Bif_2_thetadot.to_numpy()

#count time
time_array = Bif_2_t.to_numpy()
Bif_2_t = count_time(time_array,len(Bif_2_theta))



#bifurc_1
Bif_1 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_1.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_1_t=Bif_1.iloc[:,0]
Bif_1_theta=Bif_1.iloc[:,1]
Bif_1_thetadot=Bif_1.iloc[:,2]

#to numpy
Bif_1_theta = Bif_1_theta.to_numpy()
Bif_1_thetadot = Bif_1_thetadot.to_numpy()

#count time
time_array = Bif_1_t.to_numpy()
Bif_1_t = count_time(time_array,len(Bif_1_theta))



#bifurc 5
Bif_5 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_5.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_5_t=Bif_5.iloc[:,0]
Bif_5_theta=Bif_5.iloc[:,1]
Bif_5_thetadot=Bif_5.iloc[:,2]

#to numpy
Bif_5_theta = Bif_5_theta.to_numpy()
Bif_5_thetadot = Bif_5_thetadot.to_numpy()

#count time
time_array = Bif_5_t.to_numpy()
Bif_5_t = count_time(time_array,len(Bif_5_theta))



#bifurc 6
Bif_6 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_6.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_6_t=Bif_6.iloc[:,0]
Bif_6_theta=Bif_6.iloc[:,1]
Bif_6_thetadot=Bif_6.iloc[:,2]

#to numpy
Bif_6_theta = Bif_6_theta.to_numpy()
Bif_6_thetadot = Bif_6_thetadot.to_numpy()

#count time
time_array = Bif_6_t.to_numpy()
Bif_6_t = count_time(time_array,len(Bif_6_theta))



#bifurc 7
Bif_7 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_7.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_7_t=Bif_7.iloc[:,0]
Bif_7_thetadot=Bif_7.iloc[:,1]
Bif_7_theta=Bif_7.iloc[:,2]

#to numpy
Bif_7_theta = Bif_7_theta.to_numpy()
Bif_7_thetadot = Bif_7_thetadot.to_numpy()

#count time
time_array = Bif_7_t.to_numpy()
Bif_7_t = count_time(time_array,len(Bif_7_theta))


#bifurc 8
Bif_8 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_8.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_8_t=Bif_8.iloc[:,0]
Bif_8_thetadot=Bif_8.iloc[:,1]
Bif_8_theta=Bif_8.iloc[:,2]

#to numpy
Bif_8_theta = Bif_8_theta.to_numpy()
Bif_8_thetadot = Bif_8_thetadot.to_numpy()

#count time
time_array = Bif_8_t.to_numpy()
Bif_8_t = count_time(time_array,len(Bif_8_theta))

#bifurc 9
Bif_9 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_9.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_9_t=Bif_9.iloc[:,0]
Bif_9_thetadot=Bif_9.iloc[:,1]
Bif_9_theta=Bif_9.iloc[:,2]

#to numpy
Bif_9_theta = Bif_9_theta.to_numpy()
Bif_9_thetadot = Bif_9_thetadot.to_numpy()

#count time
time_array = Bif_9_t.to_numpy()
Bif_9_t = count_time(time_array,len(Bif_9_theta))


#bifurc 10
Bif_10 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_10.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_10_t=Bif_10.iloc[:,0]
Bif_10_thetadot=Bif_10.iloc[:,1]
Bif_10_theta=Bif_10.iloc[:,2]

#to numpy
Bif_10_theta = Bif_10_theta.to_numpy()
Bif_10_thetadot = Bif_10_thetadot.to_numpy()

#count time
time_array = Bif_10_t.to_numpy()
Bif_10_t = count_time(time_array,len(Bif_10_theta))

#bifurc 11
Bif_11 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_11.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_11_t=Bif_11.iloc[:,0]
Bif_11_thetadot=Bif_11.iloc[:,1]
Bif_11_theta=Bif_11.iloc[:,2]

#to numpy
Bif_11_theta = Bif_11_theta.to_numpy()
Bif_11_thetadot = Bif_11_thetadot.to_numpy()

#count time
time_array = Bif_11_t.to_numpy()
Bif_11_t = count_time(time_array,len(Bif_11_theta))


#bifurc 12
Bif_12 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_12.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_12_t=Bif_12.iloc[:,0]
Bif_12_thetadot=Bif_12.iloc[:,1]
Bif_12_theta=Bif_12.iloc[:,2]

#to numpy
Bif_12_theta = Bif_12_theta.to_numpy()
Bif_12_thetadot = Bif_12_thetadot.to_numpy()

#count time
time_array = Bif_12_t.to_numpy()
Bif_12_t = count_time(time_array,len(Bif_12_theta))



#bifurc 13
Bif_13 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_13.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_13_t=Bif_13.iloc[:,0]
Bif_13_thetadot=Bif_13.iloc[:,1]
Bif_13_theta=Bif_13.iloc[:,2]

#to numpy
Bif_13_theta = Bif_13_theta.to_numpy()
Bif_13_thetadot = Bif_13_thetadot.to_numpy()

#count time
time_array = Bif_13_t.to_numpy()
Bif_13_t = count_time(time_array,len(Bif_13_theta))


#bifurc 14
Bif_14 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_14.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_14_t=Bif_14.iloc[:,0]
Bif_14_thetadot=Bif_14.iloc[:,1]
Bif_14_theta=Bif_14.iloc[:,2]

#to numpy
Bif_14_theta = Bif_14_theta.to_numpy()
Bif_14_thetadot = Bif_14_thetadot.to_numpy()

#count time
time_array = Bif_14_t.to_numpy()
Bif_14_t = count_time(time_array,len(Bif_14_theta))


#bifurc15
Bif_15 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_15.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_15_t=Bif_15.iloc[:,0]
Bif_15_thetadot=Bif_15.iloc[:,1]
Bif_15_theta=Bif_15.iloc[:,2]

#to numpy
Bif_15_theta = Bif_15_theta.to_numpy()
Bif_15_thetadot = Bif_15_thetadot.to_numpy()

#count time
time_array = Bif_15_t.to_numpy()
Bif_15_t = count_time(time_array,len(Bif_15_theta))


#bifurc16
Bif_16 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_16.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_16_t=Bif_16.iloc[:,0]
Bif_16_thetadot=Bif_16.iloc[:,1]
Bif_16_theta=Bif_16.iloc[:,2]

#to numpy
Bif_16_theta = Bif_16_theta.to_numpy()
Bif_16_thetadot = Bif_16_thetadot.to_numpy()

#count time
time_array = Bif_16_t.to_numpy()
Bif_16_t = count_time(time_array,len(Bif_16_theta))


#bigurc17
Bif_17 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_17.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_17_t=Bif_17.iloc[:,0]
Bif_17_thetadot=Bif_17.iloc[:,1]
Bif_17_theta=Bif_17.iloc[:,2]

#to numpy
Bif_17_theta = Bif_17_theta.to_numpy()
Bif_17_thetadot = Bif_17_thetadot.to_numpy()

#count time
time_array = Bif_17_t.to_numpy()
Bif_17_t = count_time(time_array,len(Bif_17_theta))


#bifurc18
Bif_18 = pd.read_csv("TP_Chaos/Datas_Moteur/bifurc_18.csv", delimiter=';', decimal = ",")  # Adjust delimiter if necessary

Bif_18_t=Bif_18.iloc[:,0]
Bif_18_thetadot=Bif_18.iloc[:,1]
Bif_18_theta=Bif_18.iloc[:,2]

#to numpy
Bif_18_theta = Bif_18_theta.to_numpy()
Bif_18_thetadot = Bif_18_thetadot.to_numpy()

#count time
time_array = Bif_18_t.to_numpy()
Bif_18_t = count_time(time_array,len(Bif_18_theta))


Bif_1_freq_start = 5
Bif_1_freq_end = 0.5

Bif_2_freq_start = 3
Bif_2_freq_end = 0

Bif_3_freq_start = 5
Bif_3_freq_end = 0

Bif_4_freq_start = 4
Bif_4_freq_end = 1.5

Bif_5_freq_start = 4
Bif_5_freq_end = 8

Bif_6_freq_start = 1
Bif_6_freq_end = 10

Bif_7_freq_start = 3.5
Bif_7_freq_end = 1.7

Bif_8_freq_start = 1
Bif_8_freq_end = 4

Bif_9_freq_start = 1
Bif_9_freq_end = 3

Bif_10_freq_start = 1.3
Bif_10_freq_end = 2.3

Bif_11_freq_start = 2.2
Bif_11_freq_end = 2.35

Bif_12_freq_start = 2.35
Bif_12_freq_end = 2.2

Bif_13_freq_start = 2.25
Bif_13_freq_end = 1.8

Bif_14_freq_start = 5
Bif_14_freq_end = 0.5

Bif_15_freq_start = 5
Bif_15_freq_end = 0.5

Bif_16_freq_start = 2
Bif_16_freq_end = 1

Bif_17_freq_start = 9
Bif_17_freq_end = 0

Bif_18_freq_start = 5
Bif_18_freq_end = 0

phase_value = -0.05


def poincare_section(t, theta, thetadot, freq_start, freq_end, draw_interval = 0.5, section_interval=5, phase_value=0, boo = False):
    """
    Returns the Poincaré section of the phase space and the corresponding frequencies.
    
    Parameters:
    t (array): df of time points.
    theta (array): df of angular positions.
    thetadot (array): df of angular velocities.
    freq_start (float): Starting frequency of the sweep.
    freq_end (float): End frequency of the sweep.
    section_interval (float): Time interval to take Poincaré sections (e.g., every 10 seconds).
    phase_value (float): The value of theta to use for the vertical line for the Poincaré section.

    Returns:
    freqs (array): Frequencies at which the Poincaré section is taken.
    poincare_points (list): List of points in the Poincaré section as (theta, thetadot) tuples.
    """
    # Determine total time and frequency sweep rate
    total_time = t[-1] - t[0]
    freq_sweep_rate = (freq_end - freq_start) / total_time
    
    # Calculate the time points for the Poincaré sections
    poincare_times = np.arange(t[0], t[-1], draw_interval)
    
    
    #remove the last point
    poincare_times = poincare_times[:-1]
    
    # Initialize lists for storing the results
    freqs = []
    poincare_points = []

        
    for t_poincare in poincare_times:
        # Find the frequency corresponding to the current Poincaré section time
        current_freq = freq_start + freq_sweep_rate * (t_poincare - t[0])
        
        
        
        # range of theta values betweeen t_poincare and t_poincare + section_interval
        theta_range = theta[(t >= t_poincare) & (t < t_poincare + section_interval)]
        thetadot_range = thetadot[(t >= t_poincare) & (t < t_poincare + section_interval)]
        
       
      
        #if current freq is approx. equal to 2.45 Hz
        if current_freq > 3. - abs(freq_sweep_rate)*draw_interval and current_freq < 3. + abs(freq_sweep_rate)*draw_interval and boo:
            # print("Current frequency: ", current_freq)
            # print("Theta range: ", theta_range)
            # print("Thetadot range: ", thetadot_range)
            
            ax,fig = u.create_figure_and_apply_format(figsize=(12, 6),xlabel=r"$\theta$ [rad]", ylabel=r"$\dot{\theta}$ [rad/s]")
            ax.plot(theta_range, thetadot_range, color="black")
            fig.savefig("TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_Phase.png")

        
        # Find the points in phase space that cross the vertical line (theta = phase_value)
        for i in range(1, len(theta_range)):
            
            if ((theta_range[i-1] < phase_value and theta_range[i] >= phase_value) or (theta_range[i-1] >= phase_value and theta_range[i] < phase_value)) and thetadot_range[i-1] < 0 and thetadot_range[i] < 0:
                # thetadot_interp = thetadot_range[i-1]
                # Linear interpolation to find the exact crossing point
                theta_diff = theta_range[i] - theta_range[i-1]
                time_diff = t[i] - t[i-1]
                interp_fraction = (phase_value - theta_range[i-1]) / theta_diff
                
                t_interp = t[i-1] + interp_fraction * time_diff
                thetadot_interp = thetadot_range[i-1] + interp_fraction * (thetadot_range[i] - thetadot_range[i-1])
                
                freqs.append(current_freq)
                poincare_points.append(thetadot_interp)

    return np.array(freqs), np.array(poincare_points)

#plot diffurcation diagram
def plot_bifurcation_diagram(freqs, poincare_points, xlabel, ylabel, filename, figsize=(12, 6)):
    """
    Plots a bifurcation diagram.
    
    Parameters:
    freqs (array): Array of frequencies.
    poincare_points (array): Array of Poincaré section points.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """

    #phase space
    xlabel = r"Frequency [Hz]" + " [V]"
    ylabel = r"$\dot{\theta}$" + " [V]"
    ax,fig = u.create_figure_and_apply_format(figsize,xlabel=xlabel, ylabel=ylabel)

    ax.scatter(freqs,poincare_points,color="black", marker="x", s=0.5)
    
    u.set_legend_properties(ax,fontsize=18)
    fig.savefig(filename)



section_interval = 1
draw_interval = 1

# #----Bifurcation 3-----#
# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_3.png"

# freqs, poincare_points = poincare_section(Bif_3_t, Bif_3_theta, Bif_3_thetadot, Bif_3_freq_start, Bif_3_freq_end, draw_interval,section_interval, phase_value)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


# #----Bifurcation 4-----#
# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_4.png"

# freqs, poincare_points = poincare_section(Bif_4_t, Bif_4_theta, Bif_4_thetadot, Bif_4_freq_start, Bif_4_freq_end, draw_interval,section_interval, phase_value)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


# #----Bifurcation 2-----#
# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_2.png"

# freqs, poincare_points = poincare_section(Bif_2_t, Bif_2_theta, Bif_2_thetadot, Bif_2_freq_start, Bif_2_freq_end, draw_interval,section_interval, phase_value)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


# #----Bifurcation 1-----#
# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_1.png"

# freqs, poincare_points = poincare_section(Bif_1_t, Bif_1_theta, Bif_1_thetadot, Bif_1_freq_start, Bif_1_freq_end, draw_interval,section_interval, phase_value)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

# #----Bifurcation 5,6 et 7-----#
# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_5.png"

# freqs, poincare_points = poincare_section(Bif_5_t, Bif_5_theta, Bif_5_thetadot, Bif_5_freq_start, Bif_5_freq_end, draw_interval,section_interval, phase_value)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_6.png"

# freqs, poincare_points = poincare_section(Bif_6_t, Bif_6_theta, Bif_6_thetadot, Bif_6_freq_start, Bif_6_freq_end, draw_interval,section_interval, phase_value)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

# filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_7.png"

# freqs, poincare_points = poincare_section(Bif_7_t, Bif_7_theta, Bif_7_thetadot, Bif_7_freq_start, Bif_7_freq_end, draw_interval,section_interval, phase_value,True)
# plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

#----Bifurcation 8 et 9-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_8.png"

freqs, poincare_points = poincare_section(Bif_8_t, Bif_8_theta, Bif_8_thetadot, Bif_8_freq_start, Bif_8_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_9.png"

freqs, poincare_points = poincare_section(Bif_9_t, Bif_9_theta, Bif_9_thetadot, Bif_9_freq_start, Bif_9_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


#----Bifurcation 10 et 11-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_10.png"

freqs, poincare_points = poincare_section(Bif_10_t, Bif_10_theta, Bif_10_thetadot, Bif_10_freq_start, Bif_10_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_11.png"

freqs, poincare_points = poincare_section(Bif_11_t, Bif_11_theta, Bif_11_thetadot, Bif_11_freq_start, Bif_11_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


#----Bifurcation 12-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_12.png"

freqs, poincare_points = poincare_section(Bif_12_t, Bif_12_theta, Bif_12_thetadot, Bif_12_freq_start, Bif_12_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

#----Bifurcation 13-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_13.png"

freqs, poincare_points = poincare_section(Bif_13_t, Bif_13_theta, Bif_13_thetadot, Bif_13_freq_start, Bif_13_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

#----Bifurcation 14-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_14.png"

freqs, poincare_points = poincare_section(Bif_14_t, Bif_14_theta, Bif_14_thetadot, Bif_14_freq_start, Bif_14_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))



#----Bifurcation 15-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_15.png"

freqs, poincare_points = poincare_section(Bif_15_t, Bif_15_theta, Bif_15_thetadot, Bif_15_freq_start, Bif_15_freq_end, draw_interval,section_interval, phase_value,True)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


#----Bifurcation 16-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_16.png"

freqs, poincare_points = poincare_section(Bif_16_t, Bif_16_theta, Bif_16_thetadot, Bif_16_freq_start, Bif_16_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))

#----Bifurcation 17-----#
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_17.png"

freqs, poincare_points = poincare_section(Bif_17_t, Bif_17_theta, Bif_17_thetadot, Bif_17_freq_start, Bif_17_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))


#BIIFURCATION 18
filename="TP_Chaos/Figures/Moteur_Dipolaire_Bifurcation_18.png"

freqs, poincare_points = poincare_section(Bif_18_t, Bif_18_theta, Bif_18_thetadot, Bif_18_freq_start, Bif_18_freq_end, draw_interval,section_interval, phase_value)
plot_bifurcation_diagram(freqs, poincare_points, "Frequency [Hz]", r"$\dot{\theta}$ [V]", filename, figsize=(10, 6))
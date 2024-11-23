import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection

import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

module_name = "utils_v2"
file_path = "/workspaces/TP3/utils_v2.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Importer le module
import utils_v2 as u

import pandas as pd

#DEFINE FUNCTIONS
def landau_lifshitz_looyenga(Eps_inclusion,Eps_medium, f):
    
    return ((1 - f) * Eps_medium**(1/3) + f * Eps_inclusion**(1/3))**3


def bruggeman(Eps_inclusion, Eps_medium, f):
    
    def equation(Eps_Eff_500MHz):
        term_ethanol = (1 - f) * (Eps_medium - Eps_Eff_500MHz) / (Eps_medium + 2 * Eps_Eff_500MHz)
        term_water = f * (Eps_inclusion - Eps_Eff_500MHz) / (Eps_inclusion + 2 * Eps_Eff_500MHz)
        return term_ethanol + term_water
    
    # Initial guess for Eps_Eff_500MHz
    Eps_Eff_initial_guess = ((1 - f) * Eps_medium + f * Eps_inclusion)
    Eps_Eff_500MHz_solution = fsolve(equation, Eps_Eff_initial_guess)
    return Eps_Eff_500MHz_solution[0]


def looyenga(Eps_inclusion, Eps_medium, f):
    
    return ((1 - f) * Eps_medium**(1/3) + f * Eps_inclusion**(1/3))**3


def linear_mixing_rule(Eps_inclusion, Eps_medium, f):
    
    return (1 - f) * Eps_medium + f * Eps_inclusion


def lichtenecker(Eps_inclusion, Eps_medium, f):
    
    return np.exp(f * np.log(Eps_inclusion) + (1-f) * np.log(Eps_medium))





############################
# ------Import datas------ #
############################

ethanol_ml = 300
# --- Water + Ethanol --- #
# Ethanol 300 ml, water 0 ml
mixt_300_0 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 0 2024-Nov-08 12_09_47.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_0__freq = mixt_300_0.iloc[:,0]
mixt_300_0__EpsR = mixt_300_0.iloc[:,3]
mixt_300_0__EpsI = mixt_300_0.iloc[:,4]
mixt_300_0__Sigma = mixt_300_0.iloc[:,5]
mixt_300_0__TanD = mixt_300_0.iloc[:,6]

# Ethanol 300 ml, water 15 ml
mixt_300_15 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 15 2024-Nov-08 12_18_41.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_15__freq = mixt_300_15.iloc[:,0]
mixt_300_15__EpsR = mixt_300_15.iloc[:,3]
mixt_300_15__EpsI = mixt_300_15.iloc[:,4]
mixt_300_15__Sigma = mixt_300_15.iloc[:,5]
mixt_300_15__TanD = mixt_300_15.iloc[:,6]

# Ethanol 300 ml, water 30 ml
mixt_300_30 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 30 2024-Nov-08 12_22_34.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_30__freq = mixt_300_30.iloc[:,0]
mixt_300_30__EpsR = mixt_300_30.iloc[:,3]
mixt_300_30__EpsI = mixt_300_30.iloc[:,4]
mixt_300_30__Sigma = mixt_300_30.iloc[:,5]
mixt_300_30__TanD = mixt_300_30.iloc[:,6]

# Ethanol 300 ml, water 45 ml
mixt_300_45 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 45 2024-Nov-08 12_31_02.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_45__freq = mixt_300_45.iloc[:,0]
mixt_300_45__EpsR = mixt_300_45.iloc[:,3]
mixt_300_45__EpsI = mixt_300_45.iloc[:,4]
mixt_300_45__Sigma = mixt_300_45.iloc[:,5]
mixt_300_45__TanD = mixt_300_45.iloc[:,6]

# Ethanol 300 ml, water 60 ml
mixt_300_60 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 60 2024-Nov-08 12_34_29.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_60__freq = mixt_300_60.iloc[:,0]
mixt_300_60__EpsR = mixt_300_60.iloc[:,3]
mixt_300_60__EpsI = mixt_300_60.iloc[:,4]
mixt_300_60__Sigma = mixt_300_60.iloc[:,5]
mixt_300_60__TanD = mixt_300_60.iloc[:,6]

# Ethanol 300 ml, water 75 ml
mixt_300_75 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 75 2024-Nov-08 12_37_47.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_75__freq = mixt_300_75.iloc[:,0]
mixt_300_75__EpsR = mixt_300_75.iloc[:,3]
mixt_300_75__EpsI = mixt_300_75.iloc[:,4]
mixt_300_75__Sigma = mixt_300_75.iloc[:,5]
mixt_300_75__TanD = mixt_300_75.iloc[:,6]

# Ethanol 300 ml, water 90 ml
mixt_300_90 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 90 2024-Nov-08 12_42_39.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_90__freq = mixt_300_90.iloc[:,0]
mixt_300_90__EpsR = mixt_300_90.iloc[:,3]
mixt_300_90__EpsI = mixt_300_90.iloc[:,4]
mixt_300_90__Sigma = mixt_300_90.iloc[:,5]
mixt_300_90__TanD = mixt_300_90.iloc[:,6]

# Ethanol 300 ml, water 105 ml
mixt_300_105 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 105 2024-Nov-08 12_44_40.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_105__freq = mixt_300_105.iloc[:,0]
mixt_300_105__EpsR = mixt_300_105.iloc[:,3]
mixt_300_105__EpsI = mixt_300_105.iloc[:,4]
mixt_300_105__Sigma = mixt_300_105.iloc[:,5]
mixt_300_105__TanD = mixt_300_105.iloc[:,6]

# Ethanol 300 ml, water 120 ml
mixt_300_120 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 120 2024-Nov-08 12_46_40.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_120__freq = mixt_300_120.iloc[:,0]
mixt_300_120__EpsR = mixt_300_120.iloc[:,3]
mixt_300_120__EpsI = mixt_300_120.iloc[:,4]
mixt_300_120__Sigma = mixt_300_120.iloc[:,5]
mixt_300_120__TanD = mixt_300_120.iloc[:,6]

# Ethanol 300 ml, water 135 ml
mixt_300_135 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 135 2024-Nov-08 12_48_49.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_135__freq = mixt_300_135.iloc[:,0]
mixt_300_135__EpsR = mixt_300_135.iloc[:,3]
mixt_300_135__EpsI = mixt_300_135.iloc[:,4]
mixt_300_135__Sigma = mixt_300_135.iloc[:,5]
mixt_300_135__TanD = mixt_300_135.iloc[:,6]

# Ethanol 300 ml, water 150 ml
mixt_300_150 = pd.read_csv("TP_Micro-ondes/Datas/water/DAK 12 Water 24.2 deg.C 150 2024-Nov-08 12_53_08.txt", delimiter='\t', decimal = ".",header = 10)

mixt_300_150__freq = mixt_300_150.iloc[:,0]
mixt_300_150__EpsR = mixt_300_150.iloc[:,3]
mixt_300_150__EpsI = mixt_300_150.iloc[:,4]
mixt_300_150__Sigma = mixt_300_150.iloc[:,5]
mixt_300_150__TanD = mixt_300_150.iloc[:,6]







#mixture ethanol propanol
#Ethanol 300 ml, propanol 0 ml
mixt_eth_prop_300_0 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Ethanol 24.4 deg.C 0 2024-Nov-08 16_32_22.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_0__freq = mixt_eth_prop_300_0.iloc[:,0]
mixt_eth_prop_300_0__EpsR = mixt_eth_prop_300_0.iloc[:,3]
mixt_eth_prop_300_0__EpsI = mixt_eth_prop_300_0.iloc[:,4]

#Ethanol 300 ml, propanol 20 ml
mixt_eth_prop_300_20 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Ethanol 24.4 deg.C 20 2024-Nov-08 16_35_32.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_20__freq = mixt_eth_prop_300_20.iloc[:,0]
mixt_eth_prop_300_20__EpsR = mixt_eth_prop_300_20.iloc[:,3]
mixt_eth_prop_300_20__EpsI = mixt_eth_prop_300_20.iloc[:,4]

#Ethanol 300 ml, propanol 40 ml
mixt_eth_prop_300_40 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 40 2024-Nov-08 16_51_48.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_40__freq = mixt_eth_prop_300_40.iloc[:,0]
mixt_eth_prop_300_40__EpsR = mixt_eth_prop_300_40.iloc[:,3]
mixt_eth_prop_300_40__EpsI = mixt_eth_prop_300_40.iloc[:,4]

#Ethanol 300 ml, propanol 60 ml
mixt_eth_prop_300_60 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 60 2024-Nov-08 16_53_14.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_60__freq = mixt_eth_prop_300_60.iloc[:,0]
mixt_eth_prop_300_60__EpsR = mixt_eth_prop_300_60.iloc[:,3]
mixt_eth_prop_300_60__EpsI = mixt_eth_prop_300_60.iloc[:,4]

#Ethanol 300 ml, propanol 80 ml
mixt_eth_prop_300_80 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 80 2024-Nov-08 16_58_16.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_80__freq = mixt_eth_prop_300_80.iloc[:,0]
mixt_eth_prop_300_80__EpsR = mixt_eth_prop_300_80.iloc[:,3]
mixt_eth_prop_300_80__EpsI = mixt_eth_prop_300_80.iloc[:,4]

#Ethanol 300 ml, propanol 100 ml
mixt_eth_prop_300_100 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 100 2024-Nov-08 17_03_18.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_100__freq = mixt_eth_prop_300_100.iloc[:,0]
mixt_eth_prop_300_100__EpsR = mixt_eth_prop_300_100.iloc[:,3]
mixt_eth_prop_300_100__EpsI = mixt_eth_prop_300_100.iloc[:,4]

#Ethanol 300 ml, propanol 120 ml
mixt_eth_prop_300_120 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 120 2024-Nov-08 17_05_28.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_120__freq = mixt_eth_prop_300_120.iloc[:,0]
mixt_eth_prop_300_120__EpsR = mixt_eth_prop_300_120.iloc[:,3]
mixt_eth_prop_300_120__EpsI = mixt_eth_prop_300_120.iloc[:,4]

#Ethanol 300 ml, propanol 150 ml
mixt_eth_prop_300_150 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 150 2024-Nov-08 17_06_28.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_150__freq = mixt_eth_prop_300_150.iloc[:,0]
mixt_eth_prop_300_150__EpsR = mixt_eth_prop_300_150.iloc[:,3]
mixt_eth_prop_300_150__EpsI = mixt_eth_prop_300_150.iloc[:,4]

#Ethanol 300 ml, propanol 180 ml
mixt_eth_prop_300_180 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 180 2024-Nov-08 17_08_03.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_180__freq = mixt_eth_prop_300_180.iloc[:,0]
mixt_eth_prop_300_180__EpsR = mixt_eth_prop_300_180.iloc[:,3]
mixt_eth_prop_300_180__EpsI = mixt_eth_prop_300_180.iloc[:,4]

#Ethanol 300 ml, propanol 210
mixt_eth_prop_300_210 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water 24.4 deg.C 210 2024-Nov-08 17_09_29.txt", delimiter='\t', decimal = ".",header = 10)

mixt_eth_prop_300_210__freq = mixt_eth_prop_300_210.iloc[:,0]
mixt_eth_prop_300_210__EpsR = mixt_eth_prop_300_210.iloc[:,3]
mixt_eth_prop_300_210__EpsI = mixt_eth_prop_300_210.iloc[:,4]



#Water-Ethanol mixtures
#water 300 ml, ethanol 0 ml
mixt_water_eth_300_0 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 0 2024-Nov-15 13_38_01.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_0__freq = mixt_water_eth_300_0.iloc[:,0]
mixt_water_eth_300_0__EpsR = mixt_water_eth_300_0.iloc[:,3]
mixt_water_eth_300_0__EpsI = mixt_water_eth_300_0.iloc[:,4]

#water 300 ml, ethanol 15 ml
mixt_water_eth_300_15 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 15 2024-Nov-15 13_49_26.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_15__freq = mixt_water_eth_300_15.iloc[:,0]
mixt_water_eth_300_15__EpsR = mixt_water_eth_300_15.iloc[:,3]
mixt_water_eth_300_15__EpsI = mixt_water_eth_300_15.iloc[:,4]

#water 300 ml, ethanol 30 ml
mixt_water_eth_300_30 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 30 2024-Nov-15 13_52_10.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_30__freq = mixt_water_eth_300_30.iloc[:,0]
mixt_water_eth_300_30__EpsR = mixt_water_eth_300_30.iloc[:,3]
mixt_water_eth_300_30__EpsI = mixt_water_eth_300_30.iloc[:,4]

#water 300 ml, ethanol 50 ml
mixt_water_eth_300_50 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 50 2024-Nov-15 13_54_21.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_50__freq = mixt_water_eth_300_50.iloc[:,0]
mixt_water_eth_300_50__EpsR = mixt_water_eth_300_50.iloc[:,3]
mixt_water_eth_300_50__EpsI = mixt_water_eth_300_50.iloc[:,4]


#water 300 ml, ethanol 80 ml
mixt_water_eth_300_80 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 80 2024-Nov-15 13_56_05.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_80__freq = mixt_water_eth_300_80.iloc[:,0]
mixt_water_eth_300_80__EpsR = mixt_water_eth_300_80.iloc[:,3]
mixt_water_eth_300_80__EpsI = mixt_water_eth_300_80.iloc[:,4]

#water 300 ml, ethanol 120 ml
mixt_water_eth_300_120 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 120 2024-Nov-15 13_57_57.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_120__freq = mixt_water_eth_300_120.iloc[:,0]
mixt_water_eth_300_120__EpsR = mixt_water_eth_300_120.iloc[:,3]
mixt_water_eth_300_120__EpsI = mixt_water_eth_300_120.iloc[:,4]

#water 300 ml, ethanol 160 ml
mixt_water_eth_300_160 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 160 2024-Nov-15 14_02_52.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_160__freq = mixt_water_eth_300_160.iloc[:,0]
mixt_water_eth_300_160__EpsR = mixt_water_eth_300_160.iloc[:,3]
mixt_water_eth_300_160__EpsI = mixt_water_eth_300_160.iloc[:,4]

#water 300 ml, ethanol 200 ml
mixt_water_eth_300_200 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 200 2024-Nov-15 14_05_19.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_200__freq = mixt_water_eth_300_200.iloc[:,0]
mixt_water_eth_300_200__EpsR = mixt_water_eth_300_200.iloc[:,3]
mixt_water_eth_300_200__EpsI = mixt_water_eth_300_200.iloc[:,4]

#water 300 ml, ethanol 240 ml
mixt_water_eth_300_240 = pd.read_csv("TP_Micro-ondes/Datas/mixtures/DAK 12 Water-Ethanol 23.5 deg.C 240 2024-Nov-15 14_09_31.txt", delimiter='\t', decimal = ".",header = 10)

mixt_water_eth_300_240__freq = mixt_water_eth_300_240.iloc[:,0]
mixt_water_eth_300_240__EpsR = mixt_water_eth_300_240.iloc[:,3]
mixt_water_eth_300_240__EpsI = mixt_water_eth_300_240.iloc[:,4]





# Propanol 23.3°C
Propanol__23_3 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 23.3 deg.C 2024-Oct-18 10_10_07.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__23_3__freq = Propanol__23_3.iloc[:,0]
Propanol__23_3__EpsR = Propanol__23_3.iloc[:,3]
Propanol__23_3__EpsI = Propanol__23_3.iloc[:,4]
Propanol__23_3__Sigma = Propanol__23_3.iloc[:,5]
Propanol__23_3__TanD = Propanol__23_3.iloc[:,6]





ethanol_prop_ml = np.array([0, 20, 40, 60, 80, 100, 120, 150, 180, 210])
ethanol_prop_concentrations = ethanol_prop_ml/(ethanol_ml + ethanol_prop_ml)

ethanol_water_ml = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150])
ethanol_water_concentrations = ethanol_water_ml/(ethanol_ml + ethanol_water_ml)

water_ethanol_ml = np.array([0, 15, 30, 50, 80, 120, 160, 200, 240])
water_ethanol_concentrations = water_ethanol_ml/(water_ethanol_ml + 300)

mixt_freq = np.array([mixt_300_0__freq, mixt_300_15__freq, mixt_300_30__freq, mixt_300_45__freq, mixt_300_60__freq, mixt_300_75__freq, mixt_300_90__freq, mixt_300_105__freq, mixt_300_120__freq, mixt_300_135__freq, mixt_300_150__freq])
mixt_EpsI = np.array([mixt_300_0__EpsI, mixt_300_15__EpsI, mixt_300_30__EpsI, mixt_300_45__EpsI, mixt_300_60__EpsI, mixt_300_75__EpsI, mixt_300_90__EpsI, mixt_300_105__EpsI, mixt_300_120__EpsI, mixt_300_135__EpsI, mixt_300_150__EpsI])
mixt_EpsR = np.array([mixt_300_0__EpsR, mixt_300_15__EpsR, mixt_300_30__EpsR, mixt_300_45__EpsR, mixt_300_60__EpsR, mixt_300_75__EpsR, mixt_300_90__EpsR, mixt_300_105__EpsR, mixt_300_120__EpsR, mixt_300_135__EpsR, mixt_300_150__EpsR])

mixt_eth_prop_freq = np.array([mixt_eth_prop_300_0__freq, mixt_eth_prop_300_20__freq, mixt_eth_prop_300_40__freq, mixt_eth_prop_300_60__freq, mixt_eth_prop_300_80__freq, mixt_eth_prop_300_100__freq, mixt_eth_prop_300_120__freq, mixt_eth_prop_300_150__freq, mixt_eth_prop_300_180__freq, mixt_eth_prop_300_210__freq])
mixt_eth_prop_EpsI = np.array([mixt_eth_prop_300_0__EpsI, mixt_eth_prop_300_20__EpsI, mixt_eth_prop_300_40__EpsI, mixt_eth_prop_300_60__EpsI, mixt_eth_prop_300_80__EpsI, mixt_eth_prop_300_100__EpsI, mixt_eth_prop_300_120__EpsI, mixt_eth_prop_300_150__EpsI, mixt_eth_prop_300_180__EpsI, mixt_eth_prop_300_210__EpsI])
mixt_eth_prop_EpsR = np.array([mixt_eth_prop_300_0__EpsR, mixt_eth_prop_300_20__EpsR, mixt_eth_prop_300_40__EpsR, mixt_eth_prop_300_60__EpsR, mixt_eth_prop_300_80__EpsR, mixt_eth_prop_300_100__EpsR, mixt_eth_prop_300_120__EpsR, mixt_eth_prop_300_150__EpsR, mixt_eth_prop_300_180__EpsR, mixt_eth_prop_300_210__EpsR])


mixt_water_eth_freq = np.array([mixt_water_eth_300_0__freq, mixt_water_eth_300_15__freq, mixt_water_eth_300_30__freq, mixt_water_eth_300_50__freq, mixt_water_eth_300_80__freq, mixt_water_eth_300_120__freq, mixt_water_eth_300_160__freq, mixt_water_eth_300_200__freq, mixt_water_eth_300_240__freq])
mixt_water_eth_EpsI = np.array([mixt_water_eth_300_0__EpsI, mixt_water_eth_300_15__EpsI, mixt_water_eth_300_30__EpsI, mixt_water_eth_300_50__EpsI, mixt_water_eth_300_80__EpsI, mixt_water_eth_300_120__EpsI, mixt_water_eth_300_160__EpsI, mixt_water_eth_300_200__EpsI, mixt_water_eth_300_240__EpsI])
mixt_water_eth_EpsR = np.array([mixt_water_eth_300_0__EpsR, mixt_water_eth_300_15__EpsR, mixt_water_eth_300_30__EpsR, mixt_water_eth_300_50__EpsR, mixt_water_eth_300_80__EpsR, mixt_water_eth_300_120__EpsR, mixt_water_eth_300_160__EpsR, mixt_water_eth_300_200__EpsR, mixt_water_eth_300_240__EpsR])






#Ethanol 23.7°C
Ethanol__23_7 = pd.read_csv("TP_Micro-ondes/Datas/ethanol/DAK 12 Ethanol 23.7 deg.C 2024-Nov-08 10_22_03.txt", delimiter='\t', decimal = ".",header = 10)

Ethanol__23_7__freq = Ethanol__23_7.iloc[:,0]
Ethanol__23_7__EpsR = Ethanol__23_7.iloc[:,3]
Ethanol__23_7__EpsI = Ethanol__23_7.iloc[:,4]



#Propanol 23.3°C
Propanol__23_3 = pd.read_csv("TP_Micro-ondes/Datas/Propanol/DAK 12 Propanol 23.3 deg.C 2024-Oct-18 10_10_07.txt", delimiter='\t', decimal = ".",header = 10)

Propanol__23_3__freq = Propanol__23_3.iloc[:,0]
Propanol__23_3__EpsR = Propanol__23_3.iloc[:,3]
Propanol__23_3__EpsI = Propanol__23_3.iloc[:,4]







# Ethanol-water mixture
# freq vs EpsR 
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

for i, conc in reversed(list(enumerate(ethanol_water_concentrations))):
    freq = mixt_freq[i]
    EpsR = mixt_EpsR[i]
    EpsI = mixt_EpsI[i]
    
    ax.plot(freq, EpsR, label=f"{(1-conc)*100:3.0f}% E, {conc*100:3.0f}% W")
    
u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 4000)
fig.savefig("TP_Micro-ondes/Figures/mixtures_ethanol_water_EpsR.pdf")

xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

for i, conc in reversed(list(enumerate(ethanol_water_concentrations))):
    freq = mixt_freq[i]
    EpsR = mixt_EpsR[i]
    EpsI = mixt_EpsI[i]
    
    ax.plot(freq, EpsI, label=f"{(1-conc)*100:3.0f}% E, {conc*100:3.0f}% W")
    
u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 4000)
fig.savefig("TP_Micro-ondes/Figures/mixtures_ethanol_water_EpsI.pdf")



#plot EpsR vs concentration for different frequencies
ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Concentration [%]", ylabel=r"$\epsilon_r^{'}$")

#epsR at f = 500MHz
mixt_epsR_500MHz = []

for i in range(len(ethanol_water_concentrations)):
    mixt_EpsR_i = mixt_EpsR[i]
    mixt_freq_i = mixt_freq[i]
    
    mixt_epsR_500MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-500))])
    
mixt_epsR_500MHz = np.array(mixt_epsR_500MHz)
ax.scatter(ethanol_water_concentrations*100,mixt_epsR_500MHz,marker='x',label = "f = 500 MHz")


#epsR at f = 1000MHz
mixt_epsR_1000MHz = []

for i in range(len(ethanol_water_concentrations)):
    mixt_EpsR_i = mixt_EpsR[i]
    mixt_freq_i = mixt_freq[i]
    
    mixt_epsR_1000MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-1000))])
    
mixt_epsR_1000MHz = np.array(mixt_epsR_1000MHz)
ax.scatter(ethanol_water_concentrations*100,mixt_epsR_1000MHz,marker='x',label = "f = 1000 MHz")


#epsR at f = 1500MHz
mixt_epsR_1500MHz = []

for i in range(len(ethanol_water_concentrations)):
    mixt_EpsR_i = mixt_EpsR[i]
    mixt_freq_i = mixt_freq[i]
    
    mixt_epsR_1500MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-1500))])
    
mixt_epsR_1500MHz = np.array(mixt_epsR_1500MHz)
ax.scatter(ethanol_water_concentrations*100,mixt_epsR_1500MHz,marker='x',label = "f = 1500 MHz")


#plotting theoretical values of EpsR

concentration = np.linspace(0,1,1000,dtype=np.complex128)
#Lichtenecker’s logarithmic mixture law
#Eps = Eps1^((1-concentration)/100) + Eps2^((concentration)/100)
#Maxwell-Garnett effective medium theory
#Eps = (Eps_Ethanol * 2*concentration*(Eps_water - Eps_Ethanol) + Eps_Water + 2*Eps_Ethanol)/(2*Eps_Ethanol + Eps_Water - concentration*(Eps_Water - Eps_Ethanol))
#Linear mixing rule 
#Eps = (1-concentration)*Eps1 + concentration*Eps2

Eps_Ethanol_500MHz = mixt_300_0__EpsR[np.argmin(np.abs(mixt_300_0__freq-500))] + 1j*mixt_300_0__EpsI[np.argmin(np.abs(mixt_300_0__freq-500))]
Eps_Water_500MHz = mixt_water_eth_300_0__EpsR[np.argmin(np.abs(mixt_water_eth_300_0__freq-500))] + 1j*mixt_water_eth_300_0__EpsI[np.argmin(np.abs(mixt_water_eth_300_0__freq-500))]


Eps_Linear_500MHz = linear_mixing_rule(Eps_Water_500MHz, Eps_Ethanol_500MHz, concentration)
ax.plot(concentration*100, Eps_Linear_500MHz.real, label="Linear mixing rule, f = 500 MHz", linestyle='--')


Eps_Ethanol_1000MHz = mixt_300_0__EpsR[np.argmin(np.abs(mixt_300_0__freq-1000))] + 1j*mixt_300_0__EpsI[np.argmin(np.abs(mixt_300_0__freq-1000))]
Eps_Water_1000MHz = mixt_water_eth_300_0__EpsR[np.argmin(np.abs(mixt_water_eth_300_0__freq-1000))] + 1j*mixt_water_eth_300_0__EpsI[np.argmin(np.abs(mixt_water_eth_300_0__freq-1000))]

Eps_Linear_1000MHz = (1-concentration)*Eps_Ethanol_1000MHz + concentration*Eps_Water_1000MHz
ax.plot(concentration*100, Eps_Linear_1000MHz.real, label="Linear mixing rule, f = 1000 MHz", linestyle='--')



Eps_Ethanol_1500MHz = mixt_300_0__EpsR[np.argmin(np.abs(mixt_300_0__freq-1500))] + 1j*mixt_300_0__EpsI[np.argmin(np.abs(mixt_300_0__freq-1500))]
Eps_Water_1500MHz = mixt_water_eth_300_0__EpsR[np.argmin(np.abs(mixt_water_eth_300_0__freq-1500))] + 1j*mixt_water_eth_300_0__EpsI[np.argmin(np.abs(mixt_water_eth_300_0__freq-1500))]

Eps_Linear_1500MHz = (1-concentration)*Eps_Ethanol_1500MHz + concentration*Eps_Water_1500MHz
ax.plot(concentration*100, Eps_Linear_1500MHz.real, label="Linear mixing rule, f = 1500 MHz", linestyle='--')




# # u.set_legend_properties(ax,fontsize=25)
plt.tight_layout()

ax.set_xlim(-5,40)

u.set_legend_properties(ax,fontsize=18)
fig.savefig("TP_Micro-ondes/Figures/mixtures_ethanol_water_EpsR_freqs.pdf")




# Ethanol-Prop mixture
# freq vs EpsR 
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

for i, conc in list(enumerate(ethanol_prop_concentrations)):
    
    freq = mixt_eth_prop_freq[i]
    EpsR = mixt_eth_prop_EpsR[i]
    EpsI = mixt_eth_prop_EpsI[i]
    
    ax.plot(freq, EpsR, label=f"{conc*100:3.0f}% P, {(1-conc)*100:3.0f}% E")

# Add pure propanol
ax.plot(Propanol__23_3__freq, Propanol__23_3__EpsR, label="Pure  P", linestyle='--', color='black')

u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 3200)
fig.savefig("TP_Micro-ondes/Figures/mixtures_eth_prop_EpsR.pdf")

xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

for i, conc in list(enumerate(ethanol_prop_concentrations)):
    
    freq = mixt_eth_prop_freq[i]
    EpsR = mixt_eth_prop_EpsR[i]
    EpsI = mixt_eth_prop_EpsI[i]
    
    ax.plot(freq, EpsI, label=f"{conc*100:3.0f}% P, {(1-conc)*100:3.0f}% E")

# Add pure propanol
ax.plot(Propanol__23_3__freq, Propanol__23_3__EpsI, label="Pure P", linestyle='--', color='black')

u.set_legend_properties(ax, fontsize=18, bbox=(0.35, 0.7))
plt.tight_layout()
ax.set_xlim(0, 3200)
fig.savefig("TP_Micro-ondes/Figures/mixtures_eth_prop_EpsI.pdf")



#plot EpsR vs concentration for different frequencies
ax,fig = u.create_figure_and_apply_format((8,6),xlabel="Concentration [%]", ylabel=r"$\epsilon_r^{'}$")

#epsR at f = 800MHz
mixt_eth_prop_epsR_800MHz = []

for i in range(len(ethanol_prop_concentrations)):
    mixt_EpsR_i = mixt_eth_prop_EpsR[i]
    mixt_freq_i = mixt_eth_prop_freq[i]
    
    mixt_eth_prop_epsR_800MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-800))])
    
mixt_eth_prop_epsR_800MHz = np.array(mixt_eth_prop_epsR_800MHz)
ax.scatter(ethanol_prop_concentrations*100,mixt_eth_prop_epsR_800MHz,marker='x',label = "f = 800 MHz")

#epsR at f = 1000MHz
mixt_eth_prop_epsR_1000MHz = []

for i in range(len(ethanol_prop_concentrations)):
    mixt_EpsR_i = mixt_eth_prop_EpsR[i]
    mixt_freq_i = mixt_eth_prop_freq[i]
    
    mixt_eth_prop_epsR_1000MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-1000))])
    
mixt_eth_prop_epsR_1000MHz = np.array(mixt_eth_prop_epsR_1000MHz)
ax.scatter(ethanol_prop_concentrations*100,mixt_eth_prop_epsR_1000MHz,marker='x',label = "f = 1000 MHz")

#epsR at f = 1200MHz
mixt_eth_prop_epsR_1200MHz = []

for i in range(len(ethanol_prop_concentrations)):
    mixt_EpsR_i = mixt_eth_prop_EpsR[i]
    mixt_freq_i = mixt_eth_prop_freq[i]
    
    mixt_eth_prop_epsR_1200MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-1200))])
    
mixt_eth_prop_epsR_1200MHz = np.array(mixt_eth_prop_epsR_1200MHz)
ax.scatter(ethanol_prop_concentrations*100,mixt_eth_prop_epsR_1200MHz,marker='x',label = "f = 1200 MHz")



# Plotting theoretical values of EpsR for ethanol-propanol mixtures

concentration = np.linspace(0, 1, 1000, dtype=np.complex128)

Eps_Ethanol_800MHz = mixt_eth_prop_300_0__EpsR[np.argmin(np.abs(mixt_eth_prop_300_0__freq-800))] + 1j*mixt_eth_prop_300_0__EpsI[np.argmin(np.abs(mixt_eth_prop_300_0__freq-800))]
Eps_Propanol_800MHz = Propanol__23_3__EpsR[np.argmin(np.abs(Propanol__23_3__freq-800))] + 1j*Propanol__23_3__EpsI[np.argmin(np.abs(Propanol__23_3__freq-800))]

Eps_Linear_800MHz = linear_mixing_rule(Eps_Propanol_800MHz, Eps_Ethanol_800MHz, concentration)
ax.plot(concentration*100, Eps_Linear_800MHz.real, label="Linear mixing rule, f = 800 MHz", linestyle='--')

Eps_Ethanol_1000MHz = mixt_eth_prop_300_0__EpsR[np.argmin(np.abs(mixt_eth_prop_300_0__freq-1000))] + 1j*mixt_eth_prop_300_0__EpsI[np.argmin(np.abs(mixt_eth_prop_300_0__freq-1000))]
Eps_Propanol_1000MHz = Propanol__23_3__EpsR[np.argmin(np.abs(Propanol__23_3__freq-1000))] + 1j*Propanol__23_3__EpsI[np.argmin(np.abs(Propanol__23_3__freq-1000))]

Eps_Linear_1000MHz = linear_mixing_rule(Eps_Propanol_1000MHz, Eps_Ethanol_1000MHz, concentration)
ax.plot(concentration*100, Eps_Linear_1000MHz.real, label="Linear mixing rule, f = 1000 MHz", linestyle='--')

Eps_Ethanol_1200MHz = mixt_eth_prop_300_0__EpsR[np.argmin(np.abs(mixt_eth_prop_300_0__freq-1200))] + 1j*mixt_eth_prop_300_0__EpsI[np.argmin(np.abs(mixt_eth_prop_300_0__freq-1200))]
Eps_Propanol_1200MHz = Propanol__23_3__EpsR[np.argmin(np.abs(Propanol__23_3__freq-1200))] + 1j*Propanol__23_3__EpsI[np.argmin(np.abs(Propanol__23_3__freq-1200))]

Eps_Linear_1200MHz = linear_mixing_rule(Eps_Propanol_1200MHz, Eps_Ethanol_1200MHz, concentration)
ax.plot(concentration*100, Eps_Linear_1200MHz.real, label="Linear mixing rule, f = 1200 MHz", linestyle='--')

u.set_legend_properties(ax,fontsize=18)
plt.tight_layout()
fig.savefig("TP_Micro-ondes/Figures/mixtures_eth_prop_EpsR_freqs.pdf")







# Water-Ethanol mixture
# freq vs EpsR 
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"

ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

for i, conc in reversed(list(enumerate(water_ethanol_concentrations))):
    
    freq = mixt_water_eth_freq[i]
    EpsR = mixt_water_eth_EpsR[i]
    EpsI = mixt_water_eth_EpsI[i]
    
    ax.plot(freq, EpsR, label=f"{conc*100:3.0f}% E, {(1-conc)*100:3.0f}% W")
    
u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 3200)
fig.savefig("TP_Micro-ondes/Figures/mixtures_water_ethanol_EpsR.pdf")

xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"

ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

for i, conc in reversed(list(enumerate(water_ethanol_concentrations))):
    
    freq = mixt_water_eth_freq[i]
    EpsR = mixt_water_eth_EpsR[i]
    EpsI = mixt_water_eth_EpsI[i]
    
    ax.plot(freq, EpsI, label=f"{conc*100:3.0f}% E, {(1-conc)*100:3.0f}% W")
    
u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 3200)
fig.savefig("TP_Micro-ondes/Figures/mixtures_water_ethanol_EpsI.pdf")



# plot EpsR vs concentration for different frequencies
ax, fig = u.create_figure_and_apply_format((8, 6), xlabel="Concentration [%]", ylabel=r"$\epsilon_r^{'}$")

# epsR at f = 950MHz
mixt_water_eth_epsR_950MHz = []

for i in range(len(water_ethanol_concentrations)):
    mixt_EpsR_i = mixt_water_eth_EpsR[i]
    mixt_freq_i = mixt_water_eth_freq[i]
    
    mixt_water_eth_epsR_950MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-950))])
    
mixt_water_eth_epsR_950MHz = np.array(mixt_water_eth_epsR_950MHz)
ax.scatter(water_ethanol_concentrations*100, mixt_water_eth_epsR_950MHz, marker='x', label="f = 950 MHz")

# epsR at f = 1000MHz
mixt_water_eth_epsR_1000MHz = []

for i in range(len(water_ethanol_concentrations)):
    mixt_EpsR_i = mixt_water_eth_EpsR[i]
    mixt_freq_i = mixt_water_eth_freq[i]
    
    mixt_water_eth_epsR_1000MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-1000))])
    
mixt_water_eth_epsR_1000MHz = np.array(mixt_water_eth_epsR_1000MHz)
ax.scatter(water_ethanol_concentrations*100, mixt_water_eth_epsR_1000MHz, marker='x', label="f = 1000 MHz")

# epsR at f = 1050MHz
mixt_water_eth_epsR_1050MHz = []

for i in range(len(water_ethanol_concentrations)):
    mixt_EpsR_i = mixt_water_eth_EpsR[i]
    mixt_freq_i = mixt_water_eth_freq[i]
    
    mixt_water_eth_epsR_1050MHz.append(mixt_EpsR_i[np.argmin(np.abs(mixt_freq_i-1050))])
    
mixt_water_eth_epsR_1050MHz = np.array(mixt_water_eth_epsR_1050MHz)
ax.scatter(water_ethanol_concentrations*100, mixt_water_eth_epsR_1050MHz, marker='x', label="f = 1050 MHz")



# # plotting theoretical values of EpsR
# concentration = np.linspace(0, 1, 1000, dtype=np.complex128)

# Eps_Water_500MHz = mixt_water_eth_300_0__EpsR[np.argmin(np.abs(mixt_water_eth_300_0__freq-500))] + 1j*mixt_water_eth_300_0__EpsI[np.argmin(np.abs(mixt_water_eth_300_0__freq-500))]
# Eps_Ethanol_500MHz = mixt_300_0__EpsR[np.argmin(np.abs(mixt_300_0__freq-500))] + 1j*mixt_300_0__EpsI[np.argmin(np.abs(mixt_300_0__freq-500))]

# Eps_Linear_500MHz = linear_mixing_rule(Eps_Ethanol_500MHz, Eps_Water_500MHz, concentration)
# # ax.plot(concentration*100, Eps_Linear_500MHz.real, label="Linear mixing rule, f = 500 MHz", linestyle='--')


# Eps_Water_1000MHz = mixt_water_eth_300_0__EpsR[np.argmin(np.abs(mixt_water_eth_300_0__freq-1000))] + 1j*mixt_water_eth_300_0__EpsI[np.argmin(np.abs(mixt_water_eth_300_0__freq-1000))]
# Eps_Ethanol_1000MHz = mixt_300_0__EpsR[np.argmin(np.abs(mixt_300_0__freq-1000))] + 1j*mixt_300_0__EpsI[np.argmin(np.abs(mixt_300_0__freq-1000))]

# Eps_Linear_1000MHz = linear_mixing_rule(Eps_Ethanol_1000MHz,Eps_Water_1000MHz, concentration)
# # ax.plot(concentration*100, Eps_Linear_1000MHz.real, label="Linear mixing rule, f = 1000 MHz", linestyle='--')



# Eps_Water_1500MHz = mixt_water_eth_300_0__EpsR[np.argmin(np.abs(mixt_water_eth_300_0__freq-1500))] + 1j*mixt_water_eth_300_0__EpsI[np.argmin(np.abs(mixt_water_eth_300_0__freq-1500))]
# Eps_Ethanol_1500MHz = mixt_300_0__EpsR[np.argmin(np.abs(mixt_300_0__freq-1500))] + 1j*mixt_300_0__EpsI[np.argmin(np.abs(mixt_300_0__freq-1500))]

# Eps_Linear_1500MHz = linear_mixing_rule(Eps_Ethanol_1500MHz, Eps_Water_1500MHz, concentration)
# # ax.plot(concentration*100, Eps_Linear_1500MHz.real, label="Linear mixing rule, f = 1500 MHz", linestyle='--')


plt.tight_layout()
u.set_legend_properties(ax, fontsize=18)
fig.savefig("TP_Micro-ondes/Figures/mixtures_water_ethanol_EpsR_freqs.pdf")
# Plotting ethanol-water and water-ethanol EpsI together

# Create figure and axis
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{''}$"
ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

# Plot ethanol-water EpsI
for i, conc in enumerate(ethanol_water_concentrations):
    if i % 2 == 0 or i == len(water_ethanol_concentrations)-1:
        freq = mixt_freq[i]
        EpsI = mixt_EpsI[i]
        ax.plot(freq, EpsI, label=f"{(1-conc)*100:3.0f}% E, {conc*100:3.0f}% W")

# Plot water-ethanol EpsI
for i, conc in reversed(list(enumerate(water_ethanol_concentrations))):
    if i % 2 == 0 or i == len(water_ethanol_concentrations)-1:
        freq = mixt_water_eth_freq[i]
        EpsI = mixt_water_eth_EpsI[i]
        ax.plot(freq, EpsI, label=f"{conc*100:3.0f}% E, {(1-conc)*100:3.0f}% W", linestyle='--', color='black')

# Set legend properties and save figure
u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 4000)
fig.savefig("TP_Micro-ondes/Figures/combined_ethanol_water_EpsI.pdf")


# Plotting ethanol-water and water-ethanol EpsR together

# Create figure and axis
xlabel = "Frequency [MHz]"
ylabel = r"$\epsilon_r^{'}$"
ax, fig = u.create_figure_and_apply_format((12, 6), xlabel=xlabel, ylabel=ylabel)

# Plot ethanol-water EpsR
for i, conc in enumerate(ethanol_water_concentrations):
    if i % 2 == 0 or i == len(water_ethanol_concentrations)-1:
        freq = mixt_freq[i]
        EpsR = mixt_EpsR[i]
        ax.plot(freq, EpsR, label=f"{(1-conc)*100:3.0f}% E, {conc*100:3.0f}% W")

# Plot water-ethanol EpsR
for i, conc in reversed(list(enumerate(water_ethanol_concentrations))):
    if i % 2 == 0 or i == len(water_ethanol_concentrations)-1:
        freq = mixt_water_eth_freq[i]
        EpsR = mixt_water_eth_EpsR[i]
        ax.plot(freq, EpsR, label=f"{conc*100:3.0f}% E, {(1-conc)*100:3.0f}% W", linestyle='--', color='black')

# Set legend properties and save figure
u.set_legend_properties(ax, fontsize=18)
plt.tight_layout()
ax.set_xlim(0, 4000)
fig.savefig("TP_Micro-ondes/Figures/combined_ethanol_water_EpsR.pdf")

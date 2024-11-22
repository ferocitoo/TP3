import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection

from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

from scipy.optimize import curve_fit

module_name = "utils_v2"
file_path = "/workspaces/TP3/utils_v2.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Importer le module
import utils_v2 as u

import pandas as pd

# Utilisation de curve_fit pour ajuster une droite
def linear(x, a, b):
    return a * x + b

# Résistance interne

# U en V
U = np.array([0.168, 0.368, 0.568, 0.767, 0.967, 1.166, 1.365, 1.566])
dU = 0.001
# I en mA
I = np.array([10.01, 20.02, 30.03, 40.03, 50.05, 60.00, 70.01, 80.06])
dI = 0.01

I_A = I*1e-3
dI_A = dI*1e-3

# Résistance en ohm
R = U/I_A

# Incertitude sur R
dR = (1/I_A)*(dU) + (-U/(I_A**2))*dI_A

# Plot U en fonction de I

xlabel = r"$I$ (mA)"
ylabel = r"$U$ (V)"

ax,fig = u.create_figure_and_apply_format((8,6),xlabel=xlabel, ylabel=ylabel)

popt, pcov = curve_fit(linear, I, U)
a, b = popt
da, db = np.sqrt(np.diag(pcov))

print(f"{a*1e3} +- {da*1e3}")

I_fit = np.linspace(min(I), max(I), 100)
U_fit = linear(I_fit, a, b)
ax.plot(I_fit, U_fit, color='red', label=f"Fit: U = {a:.2f} I + {b:.2f}", linestyle='--')

ax.scatter(I, U, marker='x', label=r"U(I)")

u.set_legend_properties(ax,fontsize=20)

plt.tight_layout()

fig.savefig("TP_Laser/Figures/Resistance.pdf")

# Résistance externe


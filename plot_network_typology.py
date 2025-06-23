import numpy as np
import pandapower as pp
from pandapower.pd2ppc import _pd2ppc

import matplotlib.pyplot as plt
import pandapower.plotting as plot
from matplotlib.lines import Line2D
from copy import deepcopy

# configure font
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,})

# -----------------------------------------------------
# Load test network
network = pp.networks.case24_ieee_rts()
pp.rundcpp(network)
_, ppci = _pd2ppc(network)
bus_info = ppci['bus']
branch_info = ppci['branch']

# Set randomness
rng_fixed = np.random.RandomState(0)
num_gen = 38
num_WT = 10

bus_list = np.arange(bus_info.shape[0])
gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)
WT_bus_list = rng_fixed.choice(bus_list, num_WT, replace=True)

# visualise the network and plot the location of generators, loads and wind farms
# createa a copy
net_copy = deepcopy(network)
# 1. plot the base network
fig, ax = plt.subplots(figsize=(8, 14))

plot.simple_plot(net_copy, bus_color='lightgray', line_color='black', 
                 line_width=2, trafo_size = 1.5, bus_size=2, ax=ax, show_plot=False, ext_grid_size=2.4)

# 2. get bus coordinates
bus_coords = net_copy.bus_geodata[["x", "y"]]

# 3. overlay resource icons (use scatter + offset to avoid overlap)
offset = 0.01  # slightly offset the icons to avoid overlap
gen_idx = 0
WT_idx = 0
load_idx = 0
for idx, row in bus_coords.iterrows():
    x, y = row['x'], row['y']

    # check which resources are present
    has_gen = idx in gen_bus_list
    has_wind = idx in WT_bus_list
    has_load = idx in net_copy.load['bus'].values

    icon_shift = 0
    if has_gen:
        ax.scatter(x + icon_shift, y + offset, c='peru', marker='v', s=500, label='Gas Generator' if gen_idx == 0 else "", zorder=10)
        gen_idx += 1
        icon_shift += offset
    if has_wind:
        ax.scatter(x + icon_shift, y + offset, c='lime', marker='^', s=500, label='Wind Farm' if WT_idx == 0 else "", zorder=10)
        WT_idx += 1
        icon_shift += offset
    if has_load:
        ax.scatter(x + icon_shift, y + offset, c='navy', marker='x', s=250, label='Load' if load_idx == 0 else "", zorder=10)
        load_idx += 1

ax.axis('off')
ax.legend(loc='best')
plt.tight_layout()
# save the figure
plt.savefig("network_typology.png", dpi=300, bbox_inches='tight')
plt.show()
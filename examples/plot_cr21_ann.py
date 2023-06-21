import matplotlib.pyplot as plt
import numpy as np

from pygeems.slope_disp import calc_disp_cr21_ann


fig, ax = plt.subplots()

period_slide = 0.4
height_ratio = [0.2, 0.4, 0.8, 1.6]
pgv = 20
yield_coef = np.r_[0.05:0.5:0.02]


fig, ax = plt.subplots()

for hr in height_ratio:
    disp = [calc_disp_cr21_ann(pgv, yc, period_slide, hr) for yc in yield_coef]
    ax.plot(yield_coef, disp, label=f"{hr:0.1f}")

ax.legend(title="$H_{ratio}$")
ax.grid()
ax.set(xscale="linear", xlabel="$k_y$", yscale="log", ylabel="Disp (cm)", ylim=[1, 100])

fig.savefig("test.png")

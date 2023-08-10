import numpy, scipy
import matplotlib as mpl
from matplotlib import pyplot as plt

from pyscf.lib import chkfile

params = {
        "font.size":       18,
        "axes.titlesize":  20,
        "axes.labelsize":  20,
        "legend.fontsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.subplot.wspace":0.0,
        "figure.subplot.hspace":0.0,
        "axes.spines.right": True,
        "axes.spines.top":   True,
        "xtick.direction":'in',
        "ytick.direction":'in',
        "text.usetex": True,
        "font.family": "serif",
        'text.latex.preamble': r"\usepackage{amsmath}"
}
mpl.rcParams.update(params)

colors  = ["fe4a49","2ab7ca","fed766","00cc66","8c5383"]
colors  = ["#"+color for color in colors]
colors += ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:olive", "tab:cyan"]
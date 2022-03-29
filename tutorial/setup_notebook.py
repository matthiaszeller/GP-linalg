

import sys
from pathlib import Path
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project directory in the Python path
print('- adding project root in Python path')
sys.path.insert(0, str(Path(__file__).parent.parent))

# TODO torch double not float
print('- setting default torch dtype to float64')
torch.set_default_dtype(torch.double)

# Aesthetics
# Override matplotlib defaults for nicer plots
print('- overriding matplotlib aesthetics')
sns.set(style='whitegrid')
# Make plots more latex-like looking
matplotlib.rcParams.update({
    'backend': 'ps',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{gensymb}',
})
# Increase font size
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def process_figure(filename, figures_path='figs'):
    """Prettify figure (e.g. despine and make it latex-like) and save it to file"""
    path = Path(figures_path)
    if not path.exists():
        path.mkdir()
        
    sns.despine()
    plt.tight_layout()
    filepath = path.joinpath(filename)
    plt.savefig(str(filepath) + '.pdf', bbox_inches='tight')
    plt.savefig(str(filepath) + '.eps', bbox_inches='tight')


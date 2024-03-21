import argparse  
import dataPreparationUtils as dpu
import os
from scipy.stats import gamma, lognorm, pareto
try:
	from rpy2.robjects.packages import importr
except:
    os.environ['R_HOME'] = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R')
    from rpy2.robjects.packages import importr

import rpy2.robjects as ro
import pandas as pd
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate()

importr('base')
utils = importr('utils')
try:
	importr('POT')
	importr('QRM')
	importr('MASS')
	importr('eva')
	importr('nloptr')
except:
	utils.install_packages('POT', contriburl="https://cloud.r-project.org/")
	utils.install_packages('nloptr', contriburl="https://cloud.r-project.org/")
	utils.install_packages('eva', contriburl="https://cloud.r-project.org/")
	utils.install_packages('MASS', contriburl="https://cloud.r-project.org/")
	utils.install_packages('QRM', contriburl="https://cloud.r-project.org/")
	importr('POT')
	importr('MASS')
	importr('eva')
	importr('nloptr')
	importr('QRM')
 

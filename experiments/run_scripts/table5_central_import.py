import os
import argparse  
from scipy.stats import gamma, lognorm, pareto, genpareto
from rpy2.robjects.packages import importr

import rpy2.robjects as ro
import pandas as pd
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate()

importr('base')
importr('utils')
importr('nloptr')
importr('MASS')
importr('POT')
importr('QRM')
importr('eva')
 

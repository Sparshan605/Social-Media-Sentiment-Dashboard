import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import ttest_ind, ttest_rel
from scipy import stats


tiktok_data=pd.read_excel("data/tiktok_data.xlsx",header=1)
print(tiktok_data)






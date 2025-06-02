import pandas as pd
import numpy as np

def uncertainty_sorting(x,threshold):

    threshold_diff = [(abs(x - threshold), x) for rloss in x]

    np.sort(threshold_diff)
    

    uncertainty_values = [value for _, value in threshold_diff[:2]]

    return uncertainty_values


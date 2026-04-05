# Data Analysis Assignment
# This script analyzes two datasets: Iris and Wine
# Author: AI Assistant

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine

class DataAnalyzer:
    def __init__(self, dataset_name, data, feature_names):
        self.dataset_name = dataset_name
        self.data = pd.DataFrame(data, columns=feature_names)
        self.numerical_attributes = feature_names

    def compute_summary_stats(self):
        stats = {}
        for attr in self.numerical_attributes:
            values = self.data[attr]
            stats[attr] = {
                'mean': values.mean(),
                'min': values.min(),
                'max': values.max(),
                'std': values.std(),
                'median': values.median(),
                'count': len(values)
            }
        return stats
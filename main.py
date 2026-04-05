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

    def save_stats_to_file(self, filename):
        stats = self.compute_summary_stats()
        with open(filename, 'w') as f:
            f.write(f"Summary Statistics for {self.dataset_name}\n")
            f.write("=" * 50 + "\n")
            for attr, stat in stats.items():
                f.write(f"\n{attr}:\n")
                for key, value in stat.items():
                    f.write(f"  {key}: {value:.4f}\n")

def load_datasets():
    # Load Iris dataset
    iris = load_iris()
    iris_analyzer = DataAnalyzer("Iris", iris.data, iris.feature_names)

    # Load Wine dataset
    wine = load_wine()
    wine_analyzer = DataAnalyzer("Wine", wine.data, wine.feature_names)

    return iris_analyzer, wine_analyzer

def compare_datasets(analyzer1, analyzer2):
    stats1 = analyzer1.compute_summary_stats()
    stats2 = analyzer2.compute_summary_stats()

    print("Comparison of Numerical Attributes:")
    print("=" * 50)
    for attr in set(analyzer1.numerical_attributes) & set(analyzer2.numerical_attributes):
        mean1 = stats1[attr]['mean']
        mean2 = stats2[attr]['mean']
        diff = abs(mean1 - mean2)
        print(f"{attr}: Iris mean = {mean1:.4f}, Wine mean = {mean2:.4f}, Difference = {diff:.4f}")

def main():
    print("Loading datasets...")
    iris_analyzer, wine_analyzer = load_datasets()

    print(f"\nIris Dataset - Numerical Attributes: {iris_analyzer.numerical_attributes}")
    print(f"Wine Dataset - Numerical Attributes: {wine_analyzer.numerical_attributes}")

    print("\nComputing summary statistics for Iris dataset...")
    iris_stats = iris_analyzer.compute_summary_stats()
    print("Iris Statistics:")
    for attr, stat in iris_stats.items():
        print(f"{attr}: Mean={stat['mean']:.4f}, Min={stat['min']:.4f}, Max={stat['max']:.4f}, Std={stat['std']:.4f}")

    print("\nComputing summary statistics for Wine dataset...")
    wine_stats = wine_analyzer.compute_summary_stats()
    print("Wine Statistics:")
    for attr, stat in wine_stats.items():
        print(f"{attr}: Mean={stat['mean']:.4f}, Min={stat['min']:.4f}, Max={stat['max']:.4f}, Std={stat['std']:.4f}")

    # Save stats to files
    iris_analyzer.save_stats_to_file("iris_stats.txt")
    wine_analyzer.save_stats_to_file("wine_stats.txt")

    print("\nComparing datasets...")
    compare_datasets(iris_analyzer, wine_analyzer)
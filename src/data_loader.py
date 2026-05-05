"""importing the path"""
import pandas as pd

def load_data(path):
    """load the csv file we have"""
    return pd.read_csv(path)
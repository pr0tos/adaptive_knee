import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from typing import Tuple

class DataLoader:
    @staticmethod
    def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(filepath, header=None, sep=';', decimal=',')
        time_data = data.iloc[:, 0].values
        q1_data = data.iloc[:, 1].values
        
        # Удаление дубликатов
        time_data, unique_indices = np.unique(time_data, return_index=True)
        q1_data = q1_data[unique_indices]
        
        # Сглаживание (без уменьшения длины)
        q1_data = uniform_filter1d(q1_data, size=20, mode='nearest')
        
        return time_data, q1_data
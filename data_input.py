import pandas as pd
import numpy as np
from decimal import Decimal
from data_structure import Data

def getNumpyArray(line):
    row = []
    for val in line:
        try:
            row.append(float(val))
        except:
            return row
    return row

class DataLoader():
    def __init__(self, file_name):
        file = open(file_name, 'r')
        self.data = file.read().splitlines()
        self.total_inputs = int((len(self.data)+2)/16)
        print(self.total_inputs)
    
    def get_next_timestep(self):
        curr_time = float(self.data[1].split(',')[0])
        prev_time = float(self.data[2].split(',')[0])
        data = Data(self.getY(), self.getH(), self.getStd(), curr_time, prev_time)
        self.__truncate()
        return data
     
    def getY(self):
        y_index = [3]
        y = getNumpyArray(self.data[y_index[0]].split(','))
        return np.array(y).reshape(len(y), 1)
    
    def getStd(self):
        yStdDev_index = [5]
        std_dev = getNumpyArray(self.data[yStdDev_index[0]].split(','))
        return np.array(std_dev).reshape(len(std_dev), 1)
        
    def getH(self):
        h_indices = [7, 9, 11, 13]
        rows = []
        for ind in h_indices:
            rows.append(getNumpyArray(self.data[ind].split(',')))

        return np.array(rows)

    def __truncate(self):
        self.data = self.data[16:]
    
    def get_number_time_stamps(self):
        return self.total_inputs
    
    def get_truth(self, path):
        file = open(path, 'r')
        file_data = file.read().splitlines()
        file_data.pop(1)
        file_data.pop(2)
        file_data.pop(3)
        truth = []
        for line in file_data:
            vals = line.split(',')
            truth.append([float(val) for val in vals[:-1]])
        return np.array(truth)
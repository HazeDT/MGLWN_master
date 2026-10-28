import os

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data
import pickle
# --------------------------------获取数据-----------------------------

signal_size = 1024
root = "E:\data\MCC5-THU gearbox fault diagnosis datasets\变转速"


fault_name = ['health','miss_teeth',
              'gear_pitting_H','gear_pitting_L','gear_pitting_M',
              'gear_wear_H','gear_wear_L','gear_wear_M',
              'teeth_break_and_bearing_inner_H', 'teeth_break_and_bearing_inner_L','teeth_break_and_bearing_inner_M',
              'teeth_break_and_bearing_outer_H', 'teeth_break_and_bearing_outer_L','teeth_break_and_bearing_outer_M',
              'teeth_break_H','teeth_break_L','teeth_break_M',
              'teeth_crack_H','teeth_crack_L','teeth_crack_M']

# label
label = [i for i in range(20)]

class THUGearboxRadius(object):
    num_classes = 20


    def __init__(self, data_dir,SNR, InputType):
        self.sample_length = signal_size
        self.data_dir = data_dir
        self.SNR = SNR
        self.InputType = InputType


    def edge_delete(self, data):

        for i in data:
            edge_index = i.edge_index
            new_index, edge_atrr = dropout_adj(edge_index, p=0.1)
            i.edge_index = new_index

        return data

    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            print("Please check the data path, it should be a pkl file!!")

        train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
        print(len(train_dataset),len(val_dataset))
        return train_dataset, val_dataset

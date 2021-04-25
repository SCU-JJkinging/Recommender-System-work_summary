# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
def get_ui_matrix():
        data = np.array(pd.read_csv(r'../1_data/train_data.csv'))
        print(data)
        ui  = np.zeros(shape=(6040, 2536), dtype= np.int32)
        for  i in data:
            ui [i[0]][i[2]] = 1
        save = pd.DataFrame( ui)
        print(save)
        save.to_csv('train_ui_matrix.csv',index=0,header=None)

get_ui_matrix()

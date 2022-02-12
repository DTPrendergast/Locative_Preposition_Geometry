#!/usr/bin/env python
import sys
import os.path

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# Add location of project-specific python modules to system path
sys.path.insert(0, '../libs/')

import csv
import copy
import numpy as np
import pandas as pd
import tensorflow as tf

# Global variables
RESOURCES_DIR = '/resources/'
RESPONSES_FN = RESOURCES_DIR + 'sceneID_LP_and_response.csv'
SCENE_DATA_FN = RESOURCES_DIR + 'Study3_SceneData.csv'


def main():
    responses = pd.read_csv(RESPONSES_FN)
    responses.head()



def load_data(fn):
    data = []
    with open(fn,'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        csvfile.close()
    return data


if __name__ == '__main__':
    main()

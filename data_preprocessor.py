#!/usr/bin/env python
import sys
import os.path
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# Add location of project-specific python modules to system path
sys.path.insert(0, 'libs/')

# import csv
# import copy
import numpy as np
import pandas as pd
# import tensorflow as tf

# Global variables
RESOURCES_DIR = 'resources/'
RESPONSES_FN = RESOURCES_DIR + 'sceneID_LP_and_response.csv'
SCENE_DATA_FN = RESOURCES_DIR + 'Study3_SceneData.csv'
TRAINING_EXAMPLES_FN = RESOURCES_DIR + 'training_examples.csv'
NUM_SCENES = 84
ROTATION_ANGLES = np.linspace(0, 2*np.pi, num=13)[1:-1]
LP_LIST = ['above', 'against', 'behind', 'below', 'beside', 'close', 'front', 'left', 'near', 'next', 'on', 'on top', 'over', 'right']

TRNG_EXAMPLE_HEADERS = ['Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores Vector']


def main():
    responses = pd.read_csv(RESPONSES_FN)
    print(responses.head())
    print(responses.dtypes)
    print(responses.index)
    print(responses.columns)
    scene_1_responses = responses[(responses['SceneID'] == 1) and (responses['Locative Preposition'] == 'above')]
    print(scene_1_responses.head())
    print(scene_1_responses.count)

    scene_data = pd.read_csv(SCENE_DATA_FN)
    print(scene_data.head())
    print(scene_data.dtypes)
    print(scene_data.index)
    print(scene_data.columns)

    #  Build Training Examples
    trng_examples = pd.DataFrame(columns=TRNG_EXAMPLE_HEADERS)
    for scene in list(range(1, NUM_SCENES + 1)):
        for lp in LP_LIST:
            temp_df = responses[(responses['SceneID'] == scene) and (responses['Locative Preposition'] == lp)]


    #  Scale Features

    #  Data Augmentation using Rotation

def rotate_point_about_z(pt, alpha):
    #  pt is array or tuple of floats
    #  alpha is rotation in radians about the z axis
    # x' = x*cos q - y*sin q
    # y' = x*sin q + y*cos q
    x = pt[0]
    y = pt[1]
    z = pt[2]
    x_prime = (x * np.cos(alpha)) - (y * np.sin(alpha))
    y_prime = (x * np.sin(alpha)) + (y * np.cos(alpha))
    new_pt = (x_prime, y_prime, z)

    return new_pt


def load_data(fn):
    data = []
    with open(fn, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        csvfile.close()
    return data


if __name__ == '__main__':
    main()

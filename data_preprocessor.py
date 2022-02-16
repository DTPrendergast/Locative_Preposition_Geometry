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

ROTATION_ANGLES = np.linspace(0, 2*np.pi, num=13)[1:-1]
# NUM_SCENES = 84
# LP_LIST = ['above', 'against', 'behind', 'below', 'beside', 'close', 'front', 'left', 'near', 'next', 'on', 'on top', 'over', 'right']
NUM_SCENES = 4
SCENE_LIST = list(range(1, NUM_SCENES + 1))
LP_LIST = ['above', 'against', 'behind']

TRNG_EXAMPLE_HEADERS = ['Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores Vector']

# Set pandas options
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

def main():
    responses = pd.read_csv(RESPONSES_FN)
    num_responses = responses.shape[0]
    print('Responses DF Columns:  ', responses.columns)

    scene_data = pd.read_csv(SCENE_DATA_FN)
    print(scene_data.head())
    print('Scene Data DF Columns:  ', scene_data.columns)
    print('Test: ', scene_data[scene_data['SceneID'] == 1]['Cube-X'])

    #  Calculate LoS vectors from camera to objects
    los_vectors = pd.DataFrame(columns=['SceneID', 'LoS_to_Tgt', 'LoS_to_Ref'])
    los_vectors['SceneID'] = SCENE_LIST
    for scene in SCENE_LIST:
        cam_loc, tgt_loc, ref_loc = get_point_locations(scene_data, scene)
        los_to_tgt = np.subtract(tgt_loc, cam_loc)
        los_to_ref = np.subtract(ref_loc, cam_loc)
        los_vectors[los_vectors['SceneID'] == scene]['LoS_to_Tgt'] = los_to_tgt
        los_vectors[los_vectors['SceneID'] == scene]['LoS_to_Ref'] = los_to_ref

    print(los_vectors.head())

    #  Build Training Examples
    features = []
    scores = []
    trng_examples = pd.DataFrame(columns=TRNG_EXAMPLE_HEADERS)
    for scene in SCENE_LIST:
        for lp in LP_LIST:
            temp_df = responses[(responses['SceneID'] == scene) & (responses['Locative Preposition'] == lp)]
            # print(temp_df.head())
            num_rows = temp_df.shape[0]
            temp_features = np.zeros((num_rows, 6), dtype=np.float64)

            print(temp_df.shape)



    #  Scale Features

    #  Data Augmentation using Rotation


def get_point_locations(scene_data, scene):
    cam_x = scene_data[scene_data['SceneID'] == scene]['Camera Location X'].iloc[0]
    cam_y = scene_data[scene_data['SceneID'] == scene]['Camera Location Y'].iloc[0]
    cam_z = scene_data[scene_data['SceneID'] == scene]['Camera Location Z'].iloc[0]
    tgt_x = scene_data[scene_data['SceneID'] == scene]['Sphere-X'].iloc[0]
    tgt_y = scene_data[scene_data['SceneID'] == scene]['Sphere-Y'].iloc[0]
    tgt_z = scene_data[scene_data['SceneID'] == scene]['Sphere-Z'].iloc[0]
    ref_x = scene_data[scene_data['SceneID'] == scene]['Cube-X'].iloc[0]
    ref_y = scene_data[scene_data['SceneID'] == scene]['Cube-Y'].iloc[0]
    ref_z = scene_data[scene_data['SceneID'] == scene]['Cube-Z'].iloc[0]
    cam_loc = [cam_x, cam_y, cam_z]
    tgt_loc = [tgt_x, tgt_y, tgt_z]
    ref_loc = [ref_x, ref_y, ref_z]
    return cam_loc, tgt_loc, ref_loc

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

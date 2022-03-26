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

import csv
import copy
import numpy as np
import pandas as pd
# import tensorflow as tf

# Program Option: Sparse Sim Data - Just using x-y plane as an example... Actual data was collected for cases of 'left','front', and 'front-left'.  Sparse data simulates data for 'right', 'back', and 'back-right' cases.  Dense simulated data would add data for the 'back-left', 'front-left', and 'front-right' cases.
SPARSE_SIM_DATA = False

# Program Option: Force Ideal Simulated Responses - some cases do not lend themselves to effective reflection of the scene and/or simulation of inverse locative prepositions.  For example, in the base case where the object configuration is left-up, some participants may view the 'above' locative preposition as "somewhat effective".  If we were to invert the response score to get simulated data for the 'below' locative preposition, the resulting score would be "somewhat effective" for the 'below' preposition when the objects are in the left-up configuration.  This is obviously ideal response score to "not effective at all" for the 'below' locative preposition with objects in the left-up configuration.  Otherwise, the response score is inverted from the base case data.
FORCE_IDEAL_SIM_RESPONSE = True

# Program Option:  Use Bad Data - True means that program will use the respondent data that was clearly spam.  Otherwise those datasets are ignored.
USE_BAD_DATA = True
BAD_RESPONDENTS = set(['R_1plgfRZU2c26sLh', 'R_1FG4Ck5ETpJNQqG', 'R_3OfbMRoJK6rmhVT', 'R_32Mg0nmnZtkJ0Qy'])

# Global variables
RESOURCES_DIR = 'resources/'
# Input files
RESPONSES_FN = RESOURCES_DIR + 'Study3_final_responses.csv'
SCENE_DATA_FN = RESOURCES_DIR + 'Study3_SceneData.csv'
# Output file
TRAINING_EXAMPLES_FN = RESOURCES_DIR + 'training_examples.csv'
if SPARSE_SIM_DATA:
    TRAINING_EXAMPLES_FN = TRAINING_EXAMPLES_FN + '_Sparse'
if FORCE_IDEAL_SIM_RESPONSE:
    TRAINING_EXAMPLES_FN = TRAINING_EXAMPLES_FN + '_ForceIdeal'
if USE_BAD_DATA:
    TRAINING_EXAMPLES_FN = TRAINING_EXAMPLES_FN + '_WithBad'
TRAINING_EXAMPLES_FN = TRAINING_EXAMPLES_FN + '.csv'

# Rotation angles to be used for data augmentation (i.e., simulated data)
ROTATION_ANGLES = np.linspace(0, 2*np.pi, num=13)[1:-1]

# Number of scenes used for Amazon Mechanical Turk data collection
# NUM_SCENES = 84
NUM_SCENES = 4
SCENE_LIST = list(range(1, NUM_SCENES + 1))

# List of locative prepositions used in Mechanical Turk Survey
# LP_LIST = ['above', 'against', 'behind', 'below', 'beside', 'close', 'front', 'left', 'near', 'next', 'on', 'on top', 'over', 'right']
LP_LIST = ['above', 'against', 'behind']

# Headers for training examples array/dictionary/dataframe
PROCESSED_DATA_HEADERS = ['ParticipantID', 'Real/Sim', 'SceneID', 'Object Configuration', 'Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores Vector']


# Set pandas options
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def main():
    # Load responses from MTurk survey
    responses = pd.read_csv(RESPONSES_FN)
    num_participants = responses.shape[0]
    num_columns = responses.shape[1]
    print("Num participants = ", num_participants, '          num_columns = ', num_columns)
    print('Responses DF Columns:  ', responses.columns)
    print(responses.head())

    # Load scene data for the scenes presented to MTurk respondents
    scene_data = pd.read_csv(SCENE_DATA_FN)
    print('Scene Data DF Columns:  ', scene_data.columns)
    print(scene_data.head())

    #  Calculate Line-of-Sight (LoS) vectors from camera to objects.  This dataframe has one row per scene.  Each row gives vector to ref and vector to tgt.
    los_vectors = calculate_LoS_vectors_df(scene_data)
    print('los_vectors head:  ', los_vectors.head())

    # Create empty DataFrame of preprocessed data
    processed_data = pd.DataFrame(columns=PROCESSED_DATA_HEADERS)




    #  Scale Features

    #  Data Augmentation using Rotation


def calculate_LoS_vectors_df(scene_data):
    los_vectors = pd.DataFrame(columns=['SceneID', 'LoS_to_Tgt_X', 'LoS_to_Tgt_Y', 'LoS_to_Tgt_Z', 'LoS_to_Ref_X', 'LoS_to_Ref_Y', 'LoS_to_Ref_Z'])
    los_vectors['SceneID'] = SCENE_LIST
    for scene in SCENE_LIST:
        cam_loc, tgt_loc, ref_loc = get_point_locations(scene_data, scene)
        los_to_tgt = np.subtract(tgt_loc, cam_loc)
        los_to_ref = np.subtract(ref_loc, cam_loc)
        los_vectors.loc[los_vectors['SceneID'] == scene, 'LoS_to_Tgt_X'] = los_to_tgt[0]
        los_vectors.loc[los_vectors['SceneID'] == scene, 'LoS_to_Tgt_Y'] = los_to_tgt[1]
        los_vectors.loc[los_vectors['SceneID'] == scene, 'LoS_to_Tgt_Z'] = los_to_tgt[2]
        los_vectors.loc[los_vectors['SceneID'] == scene, 'LoS_to_Ref_X'] = los_to_ref[0]
        los_vectors.loc[los_vectors['SceneID'] == scene, 'LoS_to_Ref_Y'] = los_to_ref[1]
        los_vectors.loc[los_vectors['SceneID'] == scene, 'LoS_to_Ref_Z'] = los_to_ref[2]
    return los_vectors


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

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

# Program Option: Force Ideal Simulated Responses - some cases do not lend themselves to effective reflection of the scene and/or simulation of inverse locative prepositions.  For example, in the base case where the object configuration is left-up, some participants may view the 'above' locative preposition as "somewhat effective".  If we were to invert the response score to get simulated data for the 'below' locative preposition, the resulting score would be "somewhat effective" for the 'below' preposition when the objects are in the left-up configuration.  This option forces the ideal response score of "not effective at all" for the 'below' locative preposition with objects in the left-up configuration.  Otherwise, the response score is inverted from the base case data.
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
# LP_LIST = ['above', 'against', 'at', 'behind', 'below', 'beside', 'close', 'front', 'left', 'near', 'next', 'on', 'on top', 'over', 'right', 'vicinity']
LP_LIST = ['above', 'against', 'behind']

# In the raw response csv header, the locative prepositions are coded as numbers.  This dictionary enumerates which prepositions correlate with each number.
LP_CODES_IN_RAW_RESPONSES = {'1': 'above', '2': 'against', '3': 'at', '4': 'beside', '5': 'front', '6': 'left', '7': 'on', '8': 'near', '9': 'close', '10': 'next', '11': 'over', '12': 'on top', '13': 'vicinity'}

# Headers for training examples array/dictionary/dataframe
PROCESSED_DATA_HEADERS = ['ResponseID', 'Real/Sim', 'SceneID', 'Object Configuration', 'Tgt Location', 'Ref Location', 'Camera Location', 'Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores Vector']





# Set pandas options
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 999999)


def main():
    # Load responses from MTurk survey as a Pandas DataFrame
    responses = pd.read_csv(RESPONSES_FN)
    num_participants = responses.shape[0]
    num_columns = responses.shape[1]
    print("Num participants = ", num_participants, '          num_columns = ', num_columns)
    print('Responses DF Columns:  ', responses.columns)
    print(responses.head())

    # Load scene data for the scenes presented to MTurk respondents as a Pandas DataFrame
    scene_data = pd.read_csv(SCENE_DATA_FN)
    print('Scene Data DF Columns:  ', scene_data.columns)
    print(scene_data.head())

    # Create output dataframe and populate with raw data in usable format(s) without augmentation
    processed_df = process_responses(responses, scene_data)

    # TODO:  Calculate inferred responses for the behind, below, and right LP questions on the raw data.
    processed_inferred_df = infer_responses(processed_df)

    # Augment data by creating simulated object configurations and extrapolating LP response scores.  Resulting DataFrame is a concatenation of processed_df and newly simulated data.
    sim_df = add_simulated_configs(processed_df)

    # Augment data by rotating the LoS vectors




    #  Calculate Line-of-Sight (LoS) vectors from camera to objects.  This dataframe has one row per scene.  Each row gives vector to ref and vector to tgt.
    # los_vectors = calculate_LoS_vectors_df(scene_data)
    # print('los_vectors head:  ', los_vectors.head())


    # Save dataframe


def add_simulated_configs(processed_df):
    # Input:  Pandas DataFrame without configurations in behind, below, or right hemispheres
    # Output:  Pandas DataFrame with simulated configurations in behind, below, and right hemispheres.

    sim_configs = []


    return sim_configs


def infer_responses(processed_df):

    return processed_df


def process_responses(responses, scene_data):
    # Input:
    # Output:  Pandas DataFrame with columns ['ResponseID', 'Real/Sim', 'SceneID', 'Object Configuration', 'Tgt Location', 'Ref Location', 'Camera Location', 'Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores Vector']
        # LP Scores Vector is a dictionary that uses the LP_LIST as keys.  Values are filled from raw data for all LPs except for behind, below, and right after completion of this method.  Those values are inferred in the following step in the main script.

    processed_data = []

    # Iterate through responses and create each line of the processed_df
    for index, response_row in responses.iterrows():
        response_id = response_row['ResponseID']
        real_sim = 'real'
        print('ResponseID = ', response_id, '     Real/Sim = ', real_sim)

        for scene_id in SCENE_LIST:
            print('     scene_id = ', scene_id)
            obj_config = get_df_value(scene_data, 'Vector', 'SceneID', scene_id)
            tgt_loc = get_df_value(scene_data, 'Sphere Location', 'SceneID', scene_id)
            ref_loc = get_df_value(scene_data, 'Cube Location', 'SceneID', scene_id)
            cam_loc = get_df_value(scene_data, 'Camera Location', 'SceneID', scene_id)
            los_to_tgt = np.subtract(tgt_loc, cam_loc)
            los_to_ref = np.subtract(ref_loc, cam_loc)
            lp_score_dict = dict.fromkeys(LP_LIST)

            # Build the LP scores vector
            for lp_code_str in LP_CODES_IN_RAW_RESPONSES.keys():
                # Get the response score from the raw response data.  The csv header, which is the DataFrame column, is coded using a string of the responseID and SceneID (e.g., "Q4.27_9" is the column that represents responses for scene 27 and locative preposition code 9, which is "close").  Response score is shifted due to mechanics of the Qualtrics survey.  A score of 36 in the raw data is a 1 on the Likert scale, and 42 in raw data is 7 on the Likert scale.
                response_data_column = 'Q4.' + str(scene_id) + '_' + lp_code_str
                raw_score = get_df_value(responses, response_data_column, 'ResponseID', response_id)
                # Convert Qualtrics score to scale from 0 to 1
                lp_score = (float(raw_score) - 36.0) / 6.0
                lp_score_dict[LP_CODES_IN_RAW_RESPONSES[lp_code_str]] = lp_score

            # Add data row to processed_data list
            temp_row = [response_id, real_sim, scene_id, obj_config, tgt_loc, ref_loc, cam_loc, los_to_tgt[0], los_to_tgt[1], los_to_tgt[2], los_to_ref[0], los_to_ref[1], los_to_ref[2], lp_score_dict]
            processed_data.append(temp_row)

    # Create Pandas DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data, columns=PROCESSED_DATA_HEADERS)
    return processed_df


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


def get_df_value(df, tgt_val_col, ref_col, ref_col_val):
    tgt_val = df[df[ref_col] == ref_col_val][tgt_val_col].iloc[0]
    return tgt_val


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

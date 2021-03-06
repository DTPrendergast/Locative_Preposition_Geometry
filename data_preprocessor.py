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
SCENE_DATA_DTYPES = {'SceneID': int, 'Vector': str, 'Camera Location X': float, 'Camera Location Y': float, 'Camera Location Z': float, 'Sphere-X': float, 'Sphere-Y': float, 'Sphere-Z': float, 'Cube-X': float, 'Cube-Y': float, 'Cube-Z': float}

# List of locative prepositions used in Mechanical Turk Survey
# LP_LIST = ['above', 'against', 'at', 'behind', 'below', 'beside', 'close', 'front', 'left', 'near', 'next', 'on', 'on top', 'over', 'right', 'vicinity']
LP_LIST = ['above', 'against', 'behind']

# In the raw response csv header, the locative prepositions are coded as numbers.  This dictionary enumerates which prepositions correlate with each number.
LP_CODES_IN_RAW_RESPONSES = {'1': 'above', '2': 'against', '3': 'at', '4': 'beside', '5': 'front', '6': 'left', '7': 'on', '8': 'near', '9': 'close', '10': 'next', '11': 'over', '12': 'on top', '13': 'vicinity'}

# Headers for training examples array/dictionary/dataframe
PROCESSED_DATA_HEADERS = ['ExampleID', 'ResponseID', 'Real/Sim', 'Simulated from Which Example', 'SceneID', 'Object Configuration', 'Tgt Location', 'Ref Location', 'Camera Location', 'LP Scores Dict']

# Set pandas options
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 999999)


def main():
    # Load responses from MTurk survey as a Pandas DataFrame
    responses = pd.read_csv(RESPONSES_FN, index_col='ResponseID')
    num_participants = responses.shape[0]
    num_columns = responses.shape[1]
    print("Num participants = ", num_participants, '          num_columns = ', num_columns)
    print('Responses DF Columns:  ', responses.columns)
    print('Data Types:  ', responses.dtypes)
    print(responses.head())

    # Load scene data for the scenes presented to MTurk respondents as a Pandas DataFrame
    scene_data = pd.read_csv(SCENE_DATA_FN, index_col='SceneID', dtype=SCENE_DATA_DTYPES)
    print('Scene Data DF Columns:  ', scene_data.columns)
    print(' Data Types:  ', scene_data.dtypes)
    print(scene_data.head())

    # Create output dataframe and populate with raw data in usable format(s) without augmentation
    processed_df = process_responses(responses, scene_data)


    # Calculate inferred responses for the behind, below, and right LP questions on the raw data.  This process does not create new training examples, it only completes the LP score dictionary for the existing training examples.
    processed_inferred_df = fill_inferred_responses(processed_df)
    # processed_inferred_df.to_csv('outputs/step1_processed_inferred.csv')
    print('processed_inferred_df Columns:  ', processed_inferred_df.columns)
    print('Data Types:  ', processed_inferred_df.dtypes)
    print(processed_inferred_df.head())

    # Augment data by creating simulated object configurations and extrapolating LP response scores.  This process creates new training examples to complete the hemispheres of possible object configurations.  Resulting DataFrame is a concatenation of processed_df and newly simulated data.
    df_with_sims = add_simulated_configs(processed_inferred_df)
    print('sim_df Columns:  ', df_with_sims.columns)
    print('Data Types:  ', df_with_sims.dtypes)
    print(df_with_sims.head())
    # df_with_sims.to_csv('outputs/step2_simulated_configs.csv')

    # Augment data by rotating the objects about the observer position (i.e., camera position)

    # Save dataframe as csv and as data structure


def add_simulated_configs(processed_df):
    # Input:  Pandas DataFrame without configurations in behind, below, or right hemispheres
    # Output:  Pandas DataFrame with simulated configurations in behind, below, and right hemispheres.

    sim = processed_df.copy()

    #  Reverse the object configurations and reverse the object config descriptor (e.g. 'left')
    new_ref_locs = sim['Tgt Location'].copy()
    new_tgt_locs = sim['Ref Location'].copy()
    new_obj_configs = sim['Object Configuration'].apply(reverse_obj_configs)
    sim['Tgt Location'] = new_tgt_locs
    sim['Ref Location'] = new_ref_locs
    sim['Object Configuration'] = new_obj_configs

    # For the sake of traceability, log the original ExampleID on which this simmed example is based
    simmed_example = sim['ExampleID'].copy()
    sim['Simulated from Which Example'] = simmed_example
    sim['Real/Sim'] = 'sim'

    # Continue numbering the unique ExampleID numbers
    example_id = sim['ExampleID'].copy() + len(sim.index)
    sim['ExampleID'] = example_id

    # TODO:  Reverse the appropriate LP scores for this sim DataFrame
    sim = infer_scores_for_reversed_configs(sim)

    df_with_sims = pd.concat([processed_df, sim])

    #    If not SPARSE_SIM_DATA:
    #        If object config is left-front:  Create training examples for left-back, right-front, and right-back

    #        If object config is left-up:  Create training examples for left-below, right-up, and right-below

    #        If object config is up-front:  Create training examples for up-back, below-front, and below-back

    return df_with_sims


def infer_scores_for_reversed_configs(df):



    return df


def fill_inferred_responses(df):
    # Input: Pandas DataFrame where each example has an LP score dictionary which does not include scores for the behind, below, and right LPs for that scene.  Pandas DataFrame has columns ['ResponseID', 'Real/Sim', 'SceneID', 'Object Configuration', 'Tgt Location', 'Ref Location', 'Camera Location', 'Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores']
    # Output:  Same data structure as input but values for behind, below, and right LP scores inserted.

    for index, row in df.iterrows():
        obj_config = row['Object Configuration']
        scores = row['LP Scores Dict']
        behind_score, below_score, right_score = infer_responses(obj_config, scores)
        scores['behind'] = behind_score
        scores['below'] = below_score
        scores['right'] = right_score

    return df


def infer_responses(obj_config, scores):
    above = scores['above']
    front = scores['front']
    left = scores['left']

    below = above
    behind = front
    right = left

    if obj_config == 'left':
        right = reverse_score(left)
    elif obj_config == 'front':
        behind = reverse_score(front)
    elif obj_config == 'up':
        below = reverse_score(above)
    elif obj_config == 'left-front':
        right = reverse_score(left)
        behind = reverse_score(front)
    elif obj_config == 'left-up':
        right = reverse_score(left)
        below = reverse_score(above)
    elif obj_config == 'up-front':
        below = reverse_score(above)
        behind = reverse_score(front)
    else:
        print('Error inferring response.  obj_config, scores = ', obj_config, scores)

    return below, behind, right


def reverse_obj_configs(config):
    new_config = None
    if config == 'front': new_config = 'back'
    elif config == 'left': new_config = 'right'
    elif config == 'left': new_config = 'right'
    elif config == 'left-front': new_config = 'right-back'
    elif config == 'left-up': new_config = 'right-below'
    elif config == 'up': new_config = 'below'
    elif config == 'up-front': new_config = 'below-back'

    if new_config is None:
        print("Error reversing object configuration")

    return new_config


def reverse_score(score):
    return 1.0 - score


def process_responses(responses, scene_data):
    # Input:
    # Output:  Pandas DataFrame with columns ['ResponseID', 'Real/Sim', 'SceneID', 'Object Configuration', 'Tgt Location', 'Ref Location', 'Camera Location', 'Vector to Target X', 'Vector to Target Y', 'Vector to Target Z', 'Vector to Reference X', 'Vector to Reference Y', 'Vector to Reference Z', 'LP Scores']
        # LP Scores Vector is a dictionary that uses the LP_LIST as keys.  Values are filled from raw data for all LPs except for behind, below, and right after completion of this method.  Those values are inferred in the following step in the main script.

    processed_data = []

    # Iterate through responses and create each line of the processed_df
    example_id = 0
    for response_id in responses.index.values.tolist():
        # response_row = responses.iloc[[indx1]]
        # print('response_row = ', response_row)
        # response_id = responses['ResponseID'].iloc[indx1]
        real_sim = 'real'
        sim_from_response = None
        print('ResponseID = ', response_id, '     Real/Sim = ', real_sim)

        for scene_id in SCENE_LIST:
            print('     scene_id = ', scene_id)
            scene_data_row = scene_data.loc[[scene_id]]
            obj_config = scene_data['Vector'].loc[scene_id]
            # tgt_loc = scene_data['Sphere Location'].loc[scene_id]
            tgt_x = scene_data['Sphere-X'].loc[scene_id]
            tgt_y = scene_data['Sphere-Y'].loc[scene_id]
            tgt_z = scene_data['Sphere-Z'].loc[scene_id]
            tgt_loc = (float(tgt_x), float(tgt_y), float(tgt_z))
            # ref_loc = scene_data['Cube Location'].loc[scene_id]
            ref_x = scene_data['Cube-X'].loc[scene_id]
            ref_y = scene_data['Cube-Y'].loc[scene_id]
            ref_z = scene_data['Cube-Z'].loc[scene_id]
            ref_loc = (float(ref_x), float(ref_y), float(ref_z))
            # cam_loc = scene_data['Camera Location'].loc[scene_id]
            cam_x = scene_data['Camera Location X'].loc[scene_id]
            cam_y = scene_data['Camera Location Y'].loc[scene_id]
            cam_z = scene_data['Camera Location Z'].loc[scene_id]
            cam_loc = (float(cam_x), float(cam_y), float(cam_z))

            # obj_config = get_df_value(scene_data, 'Vector', 'SceneID', scene_id)
            # tgt_loc = get_df_value(scene_data, 'Sphere Location', 'SceneID', scene_id)
            # ref_loc = get_df_value(scene_data, 'Cube Location', 'SceneID', scene_id)
            # cam_loc = get_df_value(scene_data, 'Camera Location', 'SceneID', scene_id)
            # los_to_tgt = np.subtract(tgt_loc, cam_loc)
            # los_to_ref = np.subtract(ref_loc, cam_loc)
            lp_score_dict = dict.fromkeys(LP_LIST)

            # Build the LP scores vector
            for lp_code_str in LP_CODES_IN_RAW_RESPONSES.keys():
                # Get the response score from the raw response data.  The csv header, which is the DataFrame column, is coded using a string of the responseID and SceneID (e.g., "Q4.27_9" is the column that represents responses for scene 27 and locative preposition code 9, which is "close").  Response score is shifted due to mechanics of the Qualtrics survey.  A score of 36 in the raw data is a 1 on the Likert scale, and 42 in raw data is 7 on the Likert scale.
                response_data_column = 'Q4.' + str(scene_id) + '_' + lp_code_str
                raw_score = responses[response_data_column].loc[response_id]

                # Convert Qualtrics score to scale from 0 to 1
                lp_score = (float(raw_score) - 36.0) / 6.0
                lp_score_dict[LP_CODES_IN_RAW_RESPONSES[lp_code_str]] = lp_score

            # Add data row to processed_data list
            temp_row = [example_id, response_id, real_sim, sim_from_response, scene_id, obj_config, tgt_loc, ref_loc, cam_loc, lp_score_dict]
            processed_data.append(temp_row)
            example_id += 1

    # Create Pandas DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data, columns=PROCESSED_DATA_HEADERS)
    return processed_df


def get_df_value(df, tgt_val_col, ref_col, ref_col_val):
    tgt_val = df[df[ref_col] == ref_col_val][tgt_val_col].iloc[0]
    return tgt_val


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

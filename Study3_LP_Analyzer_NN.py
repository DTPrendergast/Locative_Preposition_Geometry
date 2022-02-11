import sys
import os.path

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# Add location of project-specific python modules to system path
sys.path.insert(0, '../libs/')

import csv
from math import sqrt
import numpy as np
import copy
import numpy as np
import gen_utils as gen

# Program Option:  Use Bad Data - True means that program will use the respondent data that was clearly spam.  Otherwise those datasets are ignored.
use_bad_data = False
bad_respondents = set(['R_1plgfRZU2c26sLh', 'R_1FG4Ck5ETpJNQqG', 'R_3OfbMRoJK6rmhVT', 'R_32Mg0nmnZtkJ0Qy'])


# Program Option: Sparse Sim Data - Just using x-y plane as an example... Actual data was collected for cases of 'left','front', and 'front-left'.  Sparse data simulates data for 'right', 'back', and 'back-right' cases.  Dense simulated data would add data for the 'back-left', 'front-left', and 'front-right' cases.
sparse_sim_data = True

# Program Option: Force Ideal Simulated Responses - some cases do not lend themselves to effective reflection of the scene and/or simulation of inverse locative prepositions.  For example, in the base case where the object configuration is left-up, some participants may view the 'above' locative preposition as "somewhat effective".  If we were to invert the response score to get simulated data for the 'below' locative preposition, the resulting score would be "somewhat effective" for the 'below' preposition when the objects are in the left-up configuration.  This is obviously response score to "not effective at all" for the 'below' locative preposition with objects in the left-up configuration.  Otherwise, the response score is inverted from the base case data.
force_ideal_sim_resp = True

# Program Option: Use RO as vertex - if true, the scripte uses the reference as the vertex of the two vectors when calculating the relative vector between the LoS vector and Ref-Tgt vector.  Otherwise, the midpoint of the Ref-Tgt line is used as the vertex.
use_ro_as_vertex = True

resources_dir = '../Study3-LP_Geometry/resources/'
fn_responses = resources_dir + 'Study3_final_responses.csv'
fn_scene_data = resources_dir + 'Study3_SceneData.csv'

fn_out = 'Study3_LP_Analysis'
if use_ro_as_vertex:
    fn_out = fn_out + '_ROvert'
if not sparse_sim_data:
    fn_out = fn_out + '_sparse'
if force_ideal_sim_resp:
    fn_out = fn_out + '_4sIdeal'
fn_out = fn_out + '.csv'

# num_scenes = 84

# obj_dims = (1.0, 1.0, 1.0)
# ball_vol = .5236
# cube_vol = 1.0


def main():

    resp_data = gen.read_csv_as_dict(fn_responses, None)
    print(resp_data.keys())
    num_participants = len(resp_data['ResponseId'])
    scene_data = gen.read_csv_as_dict(fn_scene_data, None)
    print(scene_data.keys())
    # resp_analysis = []

    #TODO:  Determine appropriate headers for output file
    headers = ['ParticipantID', 'Age', 'English_Language', 'Gender', 'Education', 'SceneID', 'Obj_Config', 'LP', 'Eye_Location', 'RO1_Location (cent)', 'RO1_BB_Dimensions', 'RO2_Location (cent)', 'RO2_BB_Dimensions', 'Tgt_Location (cent)', 'Tgt_BB_Dimensions', 'Response_Score']
    # resp_analysis.append(headers)

    # for col in range(cols-6):
    #     question = col + 1
    #     question_header = resp_data[0][question]
    #     test_case = ((question_header.split('.'))[1]).split('_')
    #     scene_ID = int(test_case[0])
    #     lp_code = int(test_case[1])
    #
    #     if lp_code!=20:
    #         for part in range(num_participants):
    #             resp_row = resp_data[part+1]
    #             raw_response = int(resp_row[question])
    #             if resp_row[0] not in bad_respondents:
    #                 data_rows = analyze_response(scene_data[scene_ID], resp_row, scene_ID, lp_code, raw_response)
    #                 for row in data_rows:
    #                     if row[-1]!='n/a':
    #                         resp_analysis.append(row)
    #             elif resp_row[0] in bad_respondents and use_bad_data is True:
    #                 data_rows = analyze_response(scene_data[scene_ID], resp_row, scene_ID, lp_code, raw_response)
    #                 for row in data_rows:
    #                     if row[-1]!='n/a':
    #                         resp_analysis.append(row)
    #
    # hdrs, response_dict = array_2_dict(resp_analysis)
    # features_2_norm = ['Tgt-Ref_Dist_wrt_Ref_Vol', 'Tgt-Ref_Dist_wrt_Cam-Objs_Dist']
    # normalize_features(response_dict, features_2_norm, headers)

    # # test the dictionary
    # test_row1 = []
    # test_row58 = []
    # for header in hdrs:
    #     test_row1.append(response_dict[header][0])
    #     test_row58.append(response_dict[header][113])
    #
    # print(hdrs)
    # print(resp_analysis[0])
    # print(test_row1)
    # print(resp_analysis[113])
    # print(test_row58)
    # save_2_csv(fn_out, resp_analysis)

    # save_dict_2_csv(fn_out, response_dict, headers)
    # print("done")

def normalize_features(dict, features, headers):
    for ftr in features:
        indx = headers.index(ftr)
        div = max(dict[ftr])
        new_ftr = ftr + "_(norm)"
        dict[new_ftr] = [x/div for x in dict[ftr]]
        headers.insert(indx+1, new_ftr)

def array_2_dict(array):
    dict = {}
    headers = array.pop(0)
    for key in headers:
        dict[key] = []
    for row in array:
        for indx in range(len(headers)):
            dict[headers[indx]].append(row[indx])
    return headers, dict

def load_data(fn):
    data = []
    with open(fn,'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        csvfile.close()
    return data

def filter_data(data):
    good_data = []
    for row in data:
        if row[0]!='':
            good_data.append(row)
    return good_data

def analyze_response(scene_data, resp_row, scene_ID, lp_code, raw_resp):
    data_rows = []
    LP_question = lookup_LP(lp_code)
    resp_score = shift_resp_score(raw_resp)
    cam_loc = conv_float(scene_data[3])
    scene_dims = conv_float(scene_data[2])

    # Analyze the actual scene and response data
    obj_config = scene_data[1]
    ball_loc = calc_obj_loc(conv_float(scene_data[5]))
    cube_loc = calc_obj_loc(conv_float(scene_data[9]))
    data_row = code_data_row(resp_row, scene_ID, lp_code, resp_score, obj_config, ball_loc, cube_loc, cam_loc, scene_dims,False)
    data_rows.append(data_row)
    ref_tgt_vec_wrt_LoS = data_row[12]
    x = ref_tgt_vec_wrt_LoS[0]
    y = ref_tgt_vec_wrt_LoS[1]
    z = ref_tgt_vec_wrt_LoS[2]

    # Add simulated data for 'below', 'to the right of', and 'behind' locative preposition questions
    if LP_question in set(['above','left','front']):
        data_row2 = copy.copy(data_row)
        data_row2[0:5] = ['sim','sim','sim','sim','sim']
        new_lp_question2 = rev_lp_case(LP_question)
        data_row2[7] = new_lp_question2
        data_row2[-1] = determine_resp_for_new_lp(obj_config,LP_question,resp_score,resp_score)
        data_rows.append(data_row2)

    # Create/analyze the reflected scene and response data
    obj_config3, new_ref_tgt_vec_wrt_LoS = rev_obj_config(obj_config, ref_tgt_vec_wrt_LoS)
    ball_loc3 = cube_loc
    cube_loc3 = ball_loc
    resp_score3 = determine_sim_resp_advanced(obj_config, obj_config3, LP_question, resp_score)
    data_row3 = code_data_row(resp_row, scene_ID, lp_code, resp_score3, obj_config3, ball_loc3, cube_loc3, cam_loc, scene_dims,True)
    data_row3[12] = new_ref_tgt_vec_wrt_LoS
    data_rows.append(data_row3)

    # Add simulated data for 'below', 'to the right of', and 'behind' cases using the reflected scene
    if LP_question in set(['above','left','front']):
        data_row4 = copy.copy(data_row3)
        data_row4[0:5] = ['sim','sim','sim','sim','sim']
        new_lp_question4 = rev_lp_case(LP_question)
        data_row4[7] = new_lp_question4
        data_row4[-1] = determine_resp_for_new_lp(obj_config3,LP_question,resp_score3,resp_score)
        data_rows.append(data_row4)

    if sparse_sim_data is False:
        if obj_config=="left-up":
            # Create and analyze the 'left-down' case
            obj_config5 = 'left-down'
            new_ref_tgt_vec_wrt_LoS = (x, y, -z)
            ball_loc5 = (ball_loc[0],ball_loc[1],cube_loc[2])
            cube_loc5 = (cube_loc[0],cube_loc[1],ball_loc[2])
            resp_score5 = determine_sim_resp_advanced(obj_config, obj_config5, LP_question, resp_score)
            data_row5 = code_data_row(resp_row, scene_ID, lp_code, resp_score5, obj_config5, ball_loc5, cube_loc5, cam_loc, scene_dims, True)
            data_row5[12] = new_ref_tgt_vec_wrt_LoS
            data_rows.append(data_row5)

            # Analyze the 'below', 'to the right of', and 'behind cases
            if LP_question in set(['above','left','front']):
                data_row6 = copy.copy(data_row5)
                data_row6[0:5] = ['sim','sim','sim','sim','sim']
                new_lp_question6 = rev_lp_case(LP_question)
                data_row6[7] = new_lp_question6
                data_row6[-1] = determine_resp_for_new_lp(obj_config5, LP_question, resp_score5, resp_score)
                data_rows.append(data_row6)

            # Create and analyze the 'right-up' case by reflecting the scene
            obj_config7 = 'right-up'
            new_ref_tgt_vec_wrt_LoS = (-x, y, z)
            ball_loc7 = cube_loc5
            cube_loc7 = ball_loc5
            resp_score7 = determine_sim_resp_advanced(obj_config, obj_config7, LP_question, resp_score)
            data_row7 = code_data_row(resp_row,scene_ID,lp_code,resp_score7,obj_config7,ball_loc7,cube_loc7,cam_loc,scene_dims,True)
            data_row7[12] = new_ref_tgt_vec_wrt_LoS
            data_rows.append(data_row7)

            # Analyze the 'below', 'to the right of', and 'behind' cases for the reflected scene
            if LP_question in set(['above','left','front']):
                data_row8 = copy.copy(data_row7)
                data_row8[0:5] = ['sim','sim','sim','sim','sim']
                new_lp_question8 = rev_lp_case(LP_question)
                data_row8[7] = new_lp_question8
                data_row8[-1] = determine_resp_for_new_lp(obj_config7,LP_question,resp_score7,resp_score)
                data_rows.append(data_row8)

        elif obj_config=='up-front':
            # Create and analyze the 'down-front' case
            obj_config5 = 'down-front'
            new_ref_tgt_vec_wrt_LoS = (x, y, -z)
            ball_loc5 = (ball_loc[0],ball_loc[1],cube_loc[2])
            cube_loc5 = (cube_loc[0],cube_loc[1],ball_loc[2])
            resp_score5 = determine_sim_resp_advanced(obj_config, obj_config5, LP_question, resp_score)
            data_row5 = code_data_row(resp_row,scene_ID,lp_code,resp_score5,obj_config5,ball_loc5,cube_loc5,cam_loc,scene_dims,True)
            data_row5[12] = new_ref_tgt_vec_wrt_LoS
            data_rows.append(data_row5)

            # Analyze the 'below', 'to the right of', and 'behind cases
            if LP_question in set(['above','left','front']):
                data_row6 = copy.copy(data_row5)
                data_row6[0:5] = ['sim','sim','sim','sim','sim']
                new_lp_question6 = rev_lp_case(LP_question)
                data_row6[7] = new_lp_question6
                data_row6[-1] = determine_resp_for_new_lp(obj_config5,LP_question,resp_score5,resp_score)
                data_rows.append(data_row6)

            # Create and analyze the 'right-up' case by reflecting the scene
            obj_config7 = 'up-back'
            new_ref_tgt_vec_wrt_LoS = (x, -y, z)
            ball_loc7 = cube_loc5
            cube_loc7 = ball_loc5
            resp_score7 = determine_sim_resp_advanced(obj_config, obj_config7, LP_question, resp_score)
            data_row7 = code_data_row(resp_row,scene_ID,lp_code,resp_score7,obj_config7,ball_loc7,cube_loc7,cam_loc,scene_dims,True)
            data_row7[12] = new_ref_tgt_vec_wrt_LoS
            data_rows.append(data_row7)

            # Analyze the 'below', 'to the right of', and 'behind' cases for the reflected scene
            if LP_question in set(['above','left','front']):
                data_row8 = copy.copy(data_row7)
                data_row8[0:5] = ['sim','sim','sim','sim','sim']
                new_lp_question8 = rev_lp_case(LP_question)
                data_row8[7] = new_lp_question8
                data_row8[-1] = determine_resp_for_new_lp(obj_config7,LP_question,resp_score7,resp_score)
                data_rows.append(data_row8)


        elif obj_config=='left-front':
            # Create and analyze the 'right-front' case
            obj_config5 = 'right-front'
            new_ref_tgt_vec_wrt_LoS = (-x, y, z)
            ball_loc5 = (cube_loc[0],ball_loc[1],ball_loc[2])
            cube_loc5 = (ball_loc[0],cube_loc[1],cube_loc[2])
            resp_score5 = determine_sim_resp_advanced(obj_config, obj_config5, LP_question, resp_score)
            data_row5 = code_data_row(resp_row,scene_ID,lp_code,resp_score5,obj_config5,ball_loc5,cube_loc5,cam_loc,scene_dims,True)
            data_row5[12] = new_ref_tgt_vec_wrt_LoS
            data_rows.append(data_row5)

            # Analyze the 'below', 'to the right of', and 'behind cases
            if LP_question in set(['above','left','front']):
                data_row6 = copy.copy(data_row5)
                data_row6[0:5] = ['sim','sim','sim','sim','sim']
                new_lp_question6 = rev_lp_case(LP_question)
                data_row6[7] = new_lp_question6
                data_row6[-1] = determine_resp_for_new_lp(obj_config5,LP_question,resp_score5,resp_score)
                data_rows.append(data_row6)

            # Create and analyze the 'right-up' case by reflecting the scene
            obj_config7 = 'left-back'
            new_ref_tgt_vec_wrt_LoS = (x, -y, z)
            ball_loc7 = cube_loc5
            cube_loc7 = ball_loc5
            resp_score7 = determine_sim_resp_advanced(obj_config, obj_config7, LP_question, resp_score)
            data_row7 = code_data_row(resp_row,scene_ID,lp_code,resp_score7,obj_config7,ball_loc7,cube_loc7,cam_loc,scene_dims,True)
            data_row7[12] = new_ref_tgt_vec_wrt_LoS
            data_rows.append(data_row7)

            # Analyze the 'below', 'to the right of', and 'behind' cases for the reflected scene
            if LP_question in set(['above','left','front']):
                data_row8 = copy.copy(data_row7)
                data_row8[0:5] = ['sim','sim','sim','sim','sim']
                new_lp_question8 = rev_lp_case(LP_question)
                data_row8[7] = new_lp_question8
                data_row8[-1] = determine_resp_for_new_lp(obj_config7, LP_question, resp_score7, resp_score)
                data_rows.append(data_row8)

    return data_rows

def code_data_row(resp_row, scene_ID, lp_code, resp_score, obj_config, tgt_loc, ref_loc, cam_loc,scene_dims, sim_data):
    objs_avg_loc = calc_avg_loc(tgt_loc,ref_loc)
    LP_question = lookup_LP(lp_code)

    data_row = []
    if sim_data is True:
        data_row.append('sim')
        data_row.append('sim')
        data_row.append('sim')
        data_row.append('sim')
        data_row.append('sim')
    else:
        data_row.append(resp_row[0])
        data_row.append(lookup_age(resp_row[-5]))
        data_row.append(lookup_english(resp_row[-4]))
        data_row.append(lookup_gender(resp_row[-3]))
        data_row.append(lookup_education(resp_row[-2]))
    data_row.append(scene_ID)
    data_row.append(obj_config)
    data_row.append(LP_question)
    data_row.append(ball_vol)
    data_row.append(cube_vol)

    ref_tgt_vector = calc_vector(ref_loc,tgt_loc)
    data_row.append(ref_tgt_vector)

    vertex = objs_avg_loc
    if use_ro_as_vertex:
        vertex = ref_loc
    LoS_vector = calc_vector(cam_loc,vertex)
    data_row.append(LoS_vector)

    ref_tgt_vect_wrt_LoS = calc_relative_vector(ref_tgt_vector,LoS_vector)
    data_row.append(ref_tgt_vect_wrt_LoS)

    tgt_ref_dist = calc_tgt_ref_dist(obj_config,tgt_loc,ref_loc)
    data_row.append(tgt_ref_dist)

    data_row.append(tgt_ref_dist/ball_vol)
    data_row.append(tgt_ref_dist/cube_vol)

    scene_size = calc_scene_size(scene_dims, cam_loc)
    data_row.append(scene_size)

    data_row.append(tgt_ref_dist/scene_size)

    cam_objs_dist = calc_dist(cam_loc,objs_avg_loc)
    data_row.append(cam_objs_dist)

    data_row.append(tgt_ref_dist/cam_objs_dist)

    data_row.append(resp_score)

    return data_row


def lookup_age(code):
    age_group = None
    if code=='1': age_group='18-25'
    elif code=='2': age_group='26-35'
    elif code=='3': age_group='36-45'
    elif code=='4': age_group='46-55'
    elif code=='5': age_group='56-65'
    else: print("Error determining age group.")
    return age_group

def lookup_english(code):
    if code=='1': return 'Yes'
    else: print("Error: English not participant's native language.")
    return None

def lookup_gender(code):
    gender = None
    if code=='1': gender='Male'
    elif code=='2': gender='Female'
    else: print("Error coding gender")
    return gender

def lookup_education(code):
    ed = None
    if code=='1': ed='Grade School'
    elif code=='2': ed='High School or GED'
    elif code=='3': ed='Trade/Technical/Vocational Training'
    elif code=='4': ed='Associate Degree'
    elif code=='5': ed="Bachelor's Degree"
    elif code=='6': ed='Post-Graduate/Professional Degree'
    else: print("Error determining education level.")
    return ed

def calc_scene_size(room_size, cam_pos):
    x = room_size[0]
    y = room_size[1]+(-(cam_pos[1]))
#     z = room_size[2]
    return float(x*y)

def lookup_LP(code):
    lp = None
    if code==1: lp='above'
    elif code==2: lp='against'
    elif code==3: lp='at'
    elif code==4: lp='beside'
    elif code==5: lp='front'
    elif code==6: lp='left'
    elif code==7: lp='on'
    elif code==8: lp='near'
    elif code==9: lp='close'
    elif code==10: lp='next'
    elif code==11: lp='over'
    elif code==12: lp='on top'
    elif code==13: lp='vicinity'
    else: print("Error coding the LP.")

    return lp

def calc_vector(ref,tgt):
    length = calc_dist(ref,tgt)
    x = (tgt[0]-ref[0])/length
    y = (tgt[1]-ref[1])/length
    z = (tgt[2]-ref[2])/length
    return(x,y,z)

def calc_relative_vector(obj_vector,LoS_vector):
    theta_abs = np.arctan(LoS_vector[0]/LoS_vector[1])
    theta = -theta_abs
    R = np.array([[np.cos(theta), np.sin(theta),0.0], [-(np.sin(theta)), np.cos(theta),0.0], [0.0,0.0,1.0]])
    vec_in = np.array([[obj_vector[0]],[obj_vector[1]],[obj_vector[2]]])
    vec_out = np.dot(R,vec_in)
    return (vec_out[0][0],vec_out[1][0],vec_out[2][0])

def calc_tgt_ref_dist(obj_config,tgt,ref):
    dist = None
    centroid_dist = calc_dist(tgt, ref)
    if obj_config in set(['left','front','up','right','back','down']):
        dist = centroid_dist - 1.0
    else:
        dist = centroid_dist - 1.2071

    error = abs(dist - round(dist))
    if error<.0001: dist = round(dist)

    return dist

def calc_dist(pt1, pt2):
    dist = sqrt(((pt1[0]-pt2[0])**2)+((pt1[1]-pt2[1])**2)+((pt1[2]-pt2[2])**2))
    return dist

def calc_obj_loc(obj_loc):
    x = obj_loc[0]+(obj_dims[0]/2.0)
    y = obj_loc[1]+(obj_dims[1]/2.0)
    z = obj_loc[2]+(obj_dims[2]/2.0)
    return (x,y,z)

def calc_avg_loc(pt1,pt2):
    x = (pt1[0] + pt2[0])/2
    y = (pt1[1] + pt2[1])/2
    z = (pt1[2] + pt2[2])/2
    return (x,y,z)

def conv_float(pt_string):
    pt_str = pt_string[1:-1]
    pt = pt_str.split(',')
    x = float(pt[0])
    y = float(pt[1])
    z = float(pt[2])
    return (x,y,z)

def save_2_csv(fn, data):
    with open(fn, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in range(len(data)):
            writer.writerow(data[row])
    csvfile.close()

def rev_lp_case(lp):
    rev_lp = None
    if lp=='above': rev_lp = 'below'
    elif lp=='left': rev_lp = 'right'
    elif lp=='front': rev_lp = 'behind'
    else: print("Error reversing the locative preposition case")
    return rev_lp

def rev_obj_config(obj_config, ref_tgt_vec_wrt_LoS):
    new_obj_config = None
    new_ref_tgt_vec_wrt_LoS = None
    x = ref_tgt_vec_wrt_LoS[0]
    y = ref_tgt_vec_wrt_LoS[1]
    z = ref_tgt_vec_wrt_LoS[2]
    if obj_config=='left':
        new_obj_config = 'right'
        new_ref_tgt_vec_wrt_LoS = (-x, y, z)
    elif obj_config=='front':
        new_obj_config = 'back'
        new_ref_tgt_vec_wrt_LoS = (x, -y, z)
    elif obj_config=='up':
        new_obj_config = 'down'
        new_ref_tgt_vec_wrt_LoS = (x, y, -z)
    elif obj_config=='left-front':
        new_obj_config = 'right-back'
        new_ref_tgt_vec_wrt_LoS = (-x, -y, z)
    elif obj_config=='left-up':
        new_obj_config = 'right-down'
        new_ref_tgt_vec_wrt_LoS = (-x, y, -z)
    elif obj_config=='up-front':
        new_obj_config = 'down-back'
        new_ref_tgt_vec_wrt_LoS = (x, -y, -z)
    elif obj_config=='left-down':
        new_obj_config = 'right-up'
        new_ref_tgt_vec_wrt_LoS = (-x, y, -z)
    elif obj_config=='down-front':
        new_obj_config = 'up-back'
        new_ref_tgt_vec_wrt_LoS = (x, -y, -z)
    elif obj_config=='right-front':
        new_obj_config = 'left-back'
        new_ref_tgt_vec_wrt_LoS = (-x, -y, z)
    else: print("Error reversing the object configuration string",obj_config)
    return new_obj_config, new_ref_tgt_vec_wrt_LoS

def shift_resp_score(raw_resp):
    return int(raw_resp-35)

def invert_resp_score(resp_score):
    return int(8-resp_score)

def determine_resp_for_new_lp(obj_config,LP_question,resp_score, resp_score_orig):
    new_resp_score = None
    if LP_question=='above':
        if obj_config in set(['left','right','front','back','left-front','left-back','right-back','right-front']):
            new_resp_score = resp_score
        elif obj_config in set(['up','down']):
            new_resp_score = invert_resp_score(resp_score)
        elif obj_config in set(['left-down','right-down','down-back','down-front']):
            new_resp_score = resp_score_orig
        elif obj_config in set(['left-up','right-up','up-front','up-back']):
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
    elif LP_question=='left':
        if obj_config in set(['up','down','up-front','up-back','down-back','down-front','front','back']):
            new_resp_score = resp_score
        elif obj_config in set(['left','right']):
            new_resp_score = invert_resp_score(resp_score)
        elif obj_config in set(['right-down','right-up','right-back','right-front']):
            new_resp_score = resp_score_orig
        elif obj_config in set(['left-up','left-down','left-front','left-back']):
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
    elif LP_question=='front':
        if obj_config in set(['up','down','left','right','left-up','left-down','right-down','right-up']):
            new_resp_score = resp_score
        elif obj_config in set(['front','back']):
            new_resp_score = invert_resp_score(resp_score)
        elif obj_config in set(['left-back','right-back','up-back','down-back']):
            new_resp_score = resp_score_orig
        elif obj_config in set(['left-front','right-front','up-front','down-front']):
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
    else: print("Error determining the simulated response to inverted LP.")

    return new_resp_score

def determine_sim_resp_advanced(old_obj_config,new_obj_config,lp,resp_score):
    new_resp_score = None

    if old_obj_config=='left':
        if lp=='above': new_resp_score = resp_score
        elif lp=='against': new_resp_score = resp_score
        elif lp=='at': new_resp_score = resp_score
        elif lp=='beside': new_resp_score = resp_score
        elif lp=='front': new_resp_score = resp_score
        elif lp=='left':
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
        elif lp=='on': new_resp_score = resp_score
        elif lp=='near': new_resp_score = resp_score
        elif lp=='close': new_resp_score = resp_score
        elif lp=='next': new_resp_score = resp_score
        elif lp=='over': new_resp_score = resp_score
        elif lp=='on top': new_resp_score = resp_score
        elif lp=='vicinity': new_resp_score = resp_score

    elif old_obj_config=='left-up':
        if new_obj_config=='right-down':
            if lp=='above':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front': new_resp_score = resp_score
            elif lp=='left':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on top':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='vicinity': new_resp_score = resp_score

        elif new_obj_config=='left-down':
            if lp=='above':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front': new_resp_score = resp_score
            elif lp=='left': new_resp_score = resp_score
            elif lp=='on':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on top':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='vicinity': new_resp_score = resp_score

        elif new_obj_config=='right-up':
            if lp=='above': new_resp_score = resp_score
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front': new_resp_score = resp_score
            elif lp=='left':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on': new_resp_score = resp_score
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over': new_resp_score = resp_score
            elif lp=='on top': new_resp_score = resp_score
            elif lp=='vicinity': new_resp_score = resp_score

    elif old_obj_config=='up':
        if lp=='above':
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
        elif lp=='against': new_resp_score = resp_score
        elif lp=='at': new_resp_score = resp_score
        elif lp=='beside': new_resp_score = resp_score
        elif lp=='front': new_resp_score = resp_score
        elif lp=='left': new_resp_score = resp_score
        elif lp=='on':
            if force_ideal_sim_resp is False: new_resp_score = 'n/a'
            else: new_resp_score = 1
        elif lp=='near': new_resp_score = resp_score
        elif lp=='close': new_resp_score = resp_score
        elif lp=='next': new_resp_score = resp_score
        elif lp=='over':
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
        elif lp=='on top':
            if force_ideal_sim_resp is False: new_resp_score = 'n/a'
            else: new_resp_score = 1
        elif lp=='vicinity': new_resp_score = resp_score

    elif old_obj_config=='up-front':
        if new_obj_config=='down-back':
            if lp=='above':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='left': new_resp_score = resp_score
            elif lp=='on':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on top':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='vicinity': new_resp_score = resp_score

        elif new_obj_config=='up-back':
            if lp=='above': new_resp_score = resp_score
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='left': new_resp_score = resp_score
            elif lp=='on': new_resp_score = resp_score
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over': new_resp_score = resp_score
            elif lp=='on top': new_resp_score = resp_score
            elif lp=='vicinity': new_resp_score = resp_score

        elif new_obj_config=='down-front':
            if lp=='above':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front': new_resp_score = resp_score
            elif lp=='left': new_resp_score = resp_score
            elif lp=='on':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on top':
                if force_ideal_sim_resp is False: new_resp_score = 'n/a'
                else: new_resp_score = 1
            elif lp=='vicinity': new_resp_score = resp_score


    elif old_obj_config=='front':
        if lp=='above': new_resp_score = resp_score
        elif lp=='against': new_resp_score = resp_score
        elif lp=='at': new_resp_score = resp_score
        elif lp=='beside': new_resp_score = resp_score
        elif lp=='front':
            if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
            else: new_resp_score = 1
        elif lp=='left': new_resp_score = resp_score
        elif lp=='on': new_resp_score = resp_score
        elif lp=='near': new_resp_score = resp_score
        elif lp=='close': new_resp_score = resp_score
        elif lp=='next': new_resp_score = resp_score
        elif lp=='over': new_resp_score = resp_score
        elif lp=='on top': new_resp_score = resp_score
        elif lp=='vicinity': new_resp_score = resp_score

    elif old_obj_config=='left-front':
        if new_obj_config=='right-back':
            if lp=='above': new_resp_score = resp_score
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='left':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on': new_resp_score = resp_score
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over': new_resp_score = resp_score
            elif lp=='on top': new_resp_score = resp_score
            elif lp=='vicinity': new_resp_score = resp_score

        elif new_obj_config=='right-front':
            if lp=='above': new_resp_score = resp_score
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front': new_resp_score = resp_score
            elif lp=='left':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='on': new_resp_score = resp_score
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over': new_resp_score = resp_score
            elif lp=='on top': new_resp_score = resp_score
            elif lp=='vicinity': new_resp_score = resp_score

        elif new_obj_config=='left-back':
            if lp=='above': new_resp_score = resp_score
            elif lp=='against': new_resp_score = resp_score
            elif lp=='at': new_resp_score = resp_score
            elif lp=='beside': new_resp_score = resp_score
            elif lp=='front':
                if force_ideal_sim_resp is False: new_resp_score = invert_resp_score(resp_score)
                else: new_resp_score = 1
            elif lp=='left': new_resp_score = resp_score
            elif lp=='on': new_resp_score = resp_score
            elif lp=='near': new_resp_score = resp_score
            elif lp=='close': new_resp_score = resp_score
            elif lp=='next': new_resp_score = resp_score
            elif lp=='over': new_resp_score = resp_score
            elif lp=='on top': new_resp_score = resp_score
            elif lp=='vicinity': new_resp_score = resp_score
    if new_resp_score is None:
        print("Error coding the sim response score for simulated scene data.", old_obj_config, new_obj_config, lp)
    return new_resp_score


if __name__ == '__main__':
    main()

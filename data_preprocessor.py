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
import tensorflow as tf

# Global variables
RESOURCES_DIR = '/resources/'
FN_RESPONSES = RESOURCES_DIR + 'sceneID_LP_and_response.csv'
FN_SCENE_DATA = RESOURCES_DIR + 'Study3_SceneData.csv'


def main():


if __name__ == '__main__':
    main()

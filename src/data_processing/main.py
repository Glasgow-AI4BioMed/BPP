# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import re
import sys
import time
from os.path import dirname, abspath

import torch

from divide_dataset import ReactomeDataDivider
from extract_data_form_reactome import PathWayProcessor, ReactionProcessor, PhysicalEntityProcessor, ReactomeProcessor
from py2neo import Graph, Node, Relationship
from property import Properties
from extract_data_form_reactome import FileProcessor


def extract_data_from_reactome():
    reactome_processor = ReactomeProcessor('neo4j', '123456')

    reactome_processor.execution_on_single_pathway_via_name_enhanced("Disease")

    reactome_processor.execution_on_single_pathway_via_name_enhanced("Immune System")

    reactome_processor.execution_on_single_pathway_via_name_enhanced("Metabolism")

    reactome_processor.execution_on_single_pathway_via_name_enhanced("Signal Transduction")


def divide_data_set():
    print("\033[1;36m" + "Disease" + "\033[0m" + "\n")
    disease = ReactomeDataDivider("Disease")

    disease.divide_data_for_attribute_prediction_task()
    disease.divide_data_for_input_link_prediction_task()
    disease.divide_data_for_output_link_prediction_task()

    print("\033[1;36m" + "Metabolism" + "\033[0m" + "\n")
    metabolism = ReactomeDataDivider("Metabolism")

    metabolism.divide_data_for_attribute_prediction_task()
    metabolism.divide_data_for_input_link_prediction_task()
    metabolism.divide_data_for_output_link_prediction_task()

    print("\033[1;36m" + "Immune System" + "\033[0m" + "\n")
    immune_system = ReactomeDataDivider("Immune System")

    immune_system.divide_data_for_attribute_prediction_task()
    immune_system.divide_data_for_input_link_prediction_task()
    immune_system.divide_data_for_output_link_prediction_task()

    print("\033[1;36m" + "Signal Transduction" + "\033[0m" + "\n")
    signal_transduction = ReactomeDataDivider("Signal Transduction")

    signal_transduction.divide_data_for_attribute_prediction_task()
    signal_transduction.divide_data_for_input_link_prediction_task()
    signal_transduction.divide_data_for_output_link_prediction_task()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    time_start = time.time()  # record the start time

    # extract the data from Reactome
    # extract_data_from_reactome()

    # divide the dataset
    divide_data_set()

    time_end = time.time()  # record the ending time

    time_sum = time_end - time_start  # The difference is the execution time of the program in seconds

    print("success! it takes " + str(time_sum) + " seconds to extract and process the data from Reactome")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

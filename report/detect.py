import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir + "/code")

from helpers import calculate_metrics

def detect(loader, detector):
    filename = loader.can_data[0].filename[0]
    print(f"Testing on {filename}")
    dict_of_results = detector.detect_attacks(loader)
    return filename, calculate_metrics(dict_of_results)

def detect_wrapper(args):
    return detect(args[0], args[1])
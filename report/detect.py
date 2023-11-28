import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir + "/code")

from helpers import calculate_metrics

def detect(loader, detector):
    filename = loader.can_data[0].filename[0]
    print(f"Testing on {filename}")
    return filename, calculate_metrics(detector.detect_attacks(loader))

def detect_wrapper(args):
    return detect(args[0], args[1])
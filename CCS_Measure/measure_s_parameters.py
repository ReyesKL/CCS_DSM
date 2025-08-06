import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
from lib.VST_Measurement_System import MeasurementSystem
import skrf
from pathlib import Path
import lib.vst_util_lib as util
import os

ccsURL = os.getenv("CCS_URL")

if not ccsURL:
    ccsURL = "tcp://127.0.0.1:7531/CCS.rem"

#VST SETTINGS
GATE_SOURCE_IDX = 3
DRAIN_SOURCE_IDX = 0
DRAIN_IDX_LCL = 0
GATE_IDX_LCL = 2
GATE_ANALYZER_IDX = 0
DRAIN_ANALYZER_IDX = 2
QUIESCENT_GATE = 2.1e-3  # A
QUIESCENT_DRAIN = 66e-3  # A
GATE_PINCH_OFF = -3.2
GATE_BIAS_INIT = -2.7  # V
GATE_CURRENT_LIMIT = 15e-3
GATE_MIN_MAX_V = [-4, -2.2]  # volts
DRAIN_MIN_MAX_V = [10, 28]  # volts
DRAIN_BIAS_INIT = 28  # V
DRAIN_MAX_CURRENT = 1.5  # A
DRAIN_OSC_THRESHOLD = 100e-3
GATE_STEP = 1e-2

#system characteristic impedance
Z0 = 50

# init logger
LOG_FILE_NAME = "alp_basic_test.log"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

fh = logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

VST_Sys = MeasurementSystem(ccsURL, GATE_SOURCE_IDX, DRAIN_SOURCE_IDX, GATE_IDX_LCL, DRAIN_IDX_LCL,
                            GATE_ANALYZER_IDX, DRAIN_ANALYZER_IDX, QUIESCENT_GATE, QUIESCENT_DRAIN, GATE_CURRENT_LIMIT,
                            DRAIN_OSC_THRESHOLD, DRAIN_MAX_CURRENT, GATE_MIN_MAX_V, DRAIN_MIN_MAX_V, GATE_PINCH_OFF,
                            GATE_BIAS_INIT, DRAIN_BIAS_INIT, log)

#measure the S-parameters
sparams, freqs = VST_Sys.measure_s_params(power_lvl=-10)

#create the network
network = skrf.Network(f=freqs, s=sparams, z0=Z0)

#launch the ui to save the file
file_path = util.uiputfile("Save s-parameter data", filetypes=[("Touchstone File", "*.s2p")])

#save the data
network.write_touchstone(filename=file_path)
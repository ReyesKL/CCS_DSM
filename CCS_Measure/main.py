from lib.VST_Measurement_System import MeasurementSystem
from lib.logger_setup import setup_logging
from lib.external_instrs.RTP import RTP
from lib.align_dsm import DsmAligner
import pyvisa
import numpy

"""
Initialize the VST 
"""

#URL of the VST
ccsURL = "tcp://128.138.189.140:7531/CCS.rem"

#setings for the VST
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
Z0 = 50

## Initialize the logger
log = setup_logging("test.log")


## Initialize the VST system
vst = MeasurementSystem(ccsURL, GATE_SOURCE_IDX, DRAIN_SOURCE_IDX, GATE_IDX_LCL, DRAIN_IDX_LCL,
                            GATE_ANALYZER_IDX, DRAIN_ANALYZER_IDX, QUIESCENT_GATE, QUIESCENT_DRAIN, GATE_CURRENT_LIMIT,
                            DRAIN_OSC_THRESHOLD, DRAIN_MAX_CURRENT, GATE_MIN_MAX_V, DRAIN_MIN_MAX_V, GATE_PINCH_OFF,
                            GATE_BIAS_INIT, DRAIN_BIAS_INIT, log)

rm = pyvisa.ResourceManager()

scope = RTP(rm, "TCPIP0::128.138.189.100::inst0::INSTR", log, "scope")

f0 = 2.25e9
signal_bw = 1e6
oversample_rate = 2

signal_period = 1/signal_bw
num_periods = 10

scope.set_acq_time(num_periods*signal_period)
scope.set_sample_rate(oversample_rate*(f0))
scope.get_td_data(channel=1, rerun=False, update_view=False)
# scope.auto_scale(chn=1)
# scope.get_td_data(1)

aligner = DsmAligner(scope=scope, log =log, rf_source=vst.source1, signal_period=signal_period, pa_chn=1, dsm_chn=2)
aligner.align(debug=True, atol=1e-10, n_its=10)
from statsmodels.tsa.vector_ar.plotting import plot_with_error

from lib.VST_Measurement_System import MeasurementSystem
from lib.logger_setup import setup_logging
from lib.external_instrs.RTP import RTP
from lib.align_dsm import DsmAligner
from lib.SweepVar import SweepVar, Sweep, sweep_to_xarray_from_func
import pyvisa
import numpy as np
import skrf as rf
from lib.waveform_generator import Multitone_Waveform_Generator as AWG
from lib.vst_util_lib import acpr_manager, dbm2w
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

DSM = False # true if we are running the DSM, false if PA is statically biased

signal_power = -20

#hardcoded for now, used to set up oscilloscope
f0 = 4.5e9
signal_bw = 5e6
oversample_rate = 4

signal_period = 1/signal_bw
num_periods = 2


## Initialize the logger
log = setup_logging("test.log")


## Initialize the VST system
VstSys = MeasurementSystem(ccsURL, GATE_SOURCE_IDX, DRAIN_SOURCE_IDX, GATE_IDX_LCL, DRAIN_IDX_LCL,
                            GATE_ANALYZER_IDX, DRAIN_ANALYZER_IDX, QUIESCENT_GATE, QUIESCENT_DRAIN, GATE_CURRENT_LIMIT,
                            DRAIN_OSC_THRESHOLD, DRAIN_MAX_CURRENT, GATE_MIN_MAX_V, DRAIN_MIN_MAX_V, GATE_PINCH_OFF,
                            GATE_BIAS_INIT, DRAIN_BIAS_INIT, log)

#load all tones on the measurement grid into the source (work-around at this point--just excite the entire grid)
grid_indices = VstSys.measurement_grid.index(about_center=True)
VstSys.source1.OutputEnabled = False
VstSys.source1.PlayMultitone(grid_indices, np.full_like(grid_indices,0), np.full_like(grid_indices,0))
VstSys.source2.OutputEnabled = False
VstSys.source2.PlayMultitone(grid_indices, np.full_like(grid_indices,0), np.full_like(grid_indices,0))

#create the VST tuners
VstSys.create_tuners()
source_tuner = VstSys.tuners[0]
load_tuner   = VstSys.tuners[1]

#assume that the source tuner is matched
source_tuner.gamma_0 = source_tuner.grid.full_like(0, dtype="complex")
load_tuner.gamma_0   = load_tuner.grid.full_like(0, dtype="complex")

#load signal onto source 1
awg = AWG("one_word", VstSys.measurement_grid, center_frequency=4.25e9, signal_bandwidth=10e6)
sig, par_found = awg.get_signal_with_par(8)
log.info(f"Loading signal with PAR of {par_found}dB")

#perform tuner setup with the generated signal
VstSys.setup_tuners(dbm2w(signal_power), with_signal=sig)

#have the tuner perform autoleveling 
source_tuner.move_to(gamma_des=source_tuner.gamma_0)

# source_tuner.Source.on()
acpr_calculator = acpr_manager(sig, VstSys.measurement_grid, guard_bandwidth=100e3)

rm = pyvisa.ResourceManager()
scope = RTP(rm, "TCPIP0::128.138.189.100::inst0::INSTR", log, "scope")
scope.fixture[0] = rf.Network(r"fixtures/output_fixture_dsm.s2p")
#todo initialize dc supplies for DSM
scope.set_acq_time(num_periods*signal_period)
scope.set_sample_rate(oversample_rate*(f0))

# create aligner and align the DSM and the RFPA
if DSM:
    aligner = DsmAligner(scope=scope, log =log, rf_source=vst.source1, signal_period=signal_period, pa_chn=1, dsm_chn=2)
    aligner.align(debug=True, atol=1e-10, n_its=20)

# create the sweep variables
pwr_levels = SweepVar.from_linspace("pwr_levels", -10, 10, 11)
# signals_i = SweepVar.from_list("signals", [0,1,2,3])

#create the sweep object
sweep = Sweep([pwr_levels], inner_to_outer=True)

freqs = VstSys.freqs
# measurement function to be called for each point in the sweep
def measure(pwr_levels):

    # VstSys.source1.load_signal(signals[signal_i])
    # VstSys.iteratively_set_power_level(pwr_level)

    measuredSpectra = VstSys.get_rf_data()
    a1 = measuredSpectra[0, :]
    b1 = measuredSpectra[1, :]
    a2 = measuredSpectra[2, :]
    b2 = measuredSpectra[3, :]

    pout = (np.abs(b2**2)) / 2*Z0
    pin = (np.abs(a1**2)) / 2*Z0
    pout_db = 10*np.log10(pout) + 30
    pin_db = 10*np.log10(pin) + 30
    gain_db = pout_db - pin_db

    #todo get dc data from audrey
    pdc = np.ones_like(freqs)
    pae = np.ones_like(freqs)
    acpr = np.ones_like(freqs)
    am_am = np.ones_like(freqs)
    am_pm = np.ones_like(freqs)
    # compute PAE, Pdc, Pout, Pin, Gain, IMD3/ACPR, AM_AM, AM_PM
    return {
        "PAE_p": pae,
        "Pout_db": pout_db,
        "Pin_db": pin_db,
        "Pdc_w": pdc,
        "Gain_db": gain_db,
        "ACPR_dbc": acpr,
        "am-am": am_am,
        "am-pm": am_pm
    }

output_dims = {
        "PAE_p": [("freqs", freqs)],
        "Pout_db": [("freqs", freqs)],
        "Pin_db": [("freqs", freqs)],
        "Pdc_w": [("freqs", freqs)],
        "Gain_db": [("freqs", freqs)],
        "ACPR_dbc": [("freqs", freqs)],
        "am-am": [("freqs", freqs)],
        "am-pm": [("freqs", freqs)]
    }

ds = sweep_to_xarray_from_func(sweep, measure, output_dims=output_dims)
print(ds)
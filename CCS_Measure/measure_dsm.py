from matplotlib import pyplot as plt
from lib.VST_Measurement_System import MeasurementSystem
from lib.logger_setup import setup_logging
from lib.external_instrs.RTP import RTP
from lib.external_instrs.E34401A import E34401A
from lib.align_dsm import DsmAligner
from lib.SweepVar import SweepVar, Sweep, sweep_to_xarray_from_func
import pyvisa
import numpy as np
import skrf as rf
from lib.waveform_generator import Multitone_Waveform_Generator as AWG
from lib.vst_util_lib import acpr_manager, imd3_manager, dbm2w, calc_td_powers
from lib.RKL_TOOLS import find_nearest_idx, normalize, calculate_am_xm
from lib.multitone_signal_lib import MultitoneSignal
import xarray as xr
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
voltage_lvl = 18
IS_TWO_TONE = True

# create the sweep variables
# pwr_levels = SweepVar.from_linspace("pwr_level", 1, 31, 16)
pwr_levels = SweepVar.from_list("pwr_level", list(np.arange(1,33,step=2)))
# signals_i = SweepVar.from_list("signals", [0,1,2,3])

#this is the signal power that we want to measure.
signal_power = 0

#hardcoded for now, used to set up oscilloscope
f0 = 4.5e9
signal_bw = 5e6
oversample_rate = 4
signal_period = 1/signal_bw
num_periods = 4

rf_chan = 1
dsm_chan = 3


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
VstSys.find_reflection_coefficients(source_1_power=-30, source_2_power=-30, return_power_lvl=-30, use_grid="tuner")
source_tuner.gamma_0 = source_tuner.grid.full_like(0, dtype="complex")
load_tuner.gamma_0   = load_tuner.grid.full_like(0, dtype="complex")


#load signal onto source 1
awg = AWG("one_word", VstSys.measurement_grid, center_frequency=4.25e9, signal_bandwidth=2e6)
sig, par_found = awg.get_signal_with_par(3)
log.info(f"Loading signal with PAR of {par_found:.2f}dB")
# sig = MultitoneSignal.single_tone(4.25e9, VstSys.measurement_grid, power=dbm2w(signal_power), side_tones_amp=-30)


#perform tuner setup with the generated signal
VstSys.setup_tuners(dbm2w(signal_power), with_signal=sig)

#have the tuner perform autoleveling 
source_tuner.move_to(gamma_des=source_tuner.gamma_0)

#EXPLICITLY load the SIGNAL into SOURCE 2
load_tuner.S0  = sig

# source_tuner.Source.on()
# VstSys.load_signal(sig, 1)
if IS_TWO_TONE:
    linearity_calculator = acpr_manager(sig, VstSys.measurement_grid, guard_bandwidth=100e3)
else:
    linearity_calculator = imd3_manager(sig, VstSys.measurement_grid)

 
#load the signal into source 2
source = VstSys.source1
rel_ph = np.array(list(map(float,source.RelativeMultiTones.RelativePhases))) 
rel_amp = np.array(list(map(float, source.RelativeMultiTones.RelativeAmplitudes)))
source1_power = source.OutputLevel

#now transfer this signal to the second source
sig_indices = VstSys.measurement_grid.cast_index(sig.grid, about_center=True)
source = VstSys.source2
source.OutputLevel = source1_power
source.PlayMultitone((list(sig_indices)), rel_amp, rel_ph)

#initalize the visa interface
rm = pyvisa.ResourceManager()

# initialize the scope
scope = RTP(rm, "TCPIP0::128.138.189.100::inst0::INSTR", log, "scope")
# load the output fixture into the scope object for de-embedding
scope.fixtures[0] = rf.Network(r"fixtures/output_fixture_dsm.s2p")

#setup the oscilloscope sampling and horizontal timebase
scope.set_acq_time(num_periods*signal_period)
scope.set_sample_rate(oversample_rate*(f0))


# initialize the current meters
bias_current_meter = E34401A(rm, "GPIB1::29::INSTR", log, "meter1")
main_current_meter = E34401A(rm, "GPIB1::19::INSTR", log, "meter1")
bias_current_meter.auto_range()
main_current_meter.auto_range()


#initialize the aligner
aligner = DsmAligner(scope=scope,
                     log=log,
                     rf_source=VstSys.source1,
                     signal_period=signal_period,
                     pa_chn=rf_chan,
                     dsm_chn=dsm_chan)

# turn on the rf source
source_tuner.Source.on()

# create aligner and align the DSM and the RFPA
if DSM:
    aligner.align(debug=True, atol=1e-9, n_its=20)



#create the sweep object
sweep = Sweep([pwr_levels], inner_to_outer=True)

#get freqs for creating xarrays
freqs = VstSys.freqs

# dummy again for creating xarrays
t_scope, rf_td_wavefrom  = scope.get_td_data(rf_chan)

# dummy again for creating xarrays
measuredSpectra = VstSys.get_rf_data()
a1 = measuredSpectra[0, :]
b1 = measuredSpectra[1, :]
a2 = measuredSpectra[2, :]
b2 = measuredSpectra[3, :]
_, _, t_vst = calc_td_powers(a1, b1, a2, b2, freqs)


# measurement function to be called for each point in the sweep
def measure(pwr_level):

    #todo set power level
    sig.power = dbm2w(pwr_level)
    source_tuner.move_to()
    scope.auto_scale(rf_chan)
    t_scope, rf_td_wavefrom  = scope.get_td_data(rf_chan)
    scope.auto_scale(dsm_chan)
    _, dsm_td_waveform = scope.get_td_data(dsm_chan)
    i_bias = bias_current_meter.meas_dc_current()
    i_main = main_current_meter.meas_dc_current()

    # measuredSpectra = VstSys.get_rf_data()
    a1 = VstSys.measuredSpectra[0, :]
    b1 = VstSys.measuredSpectra[1, :]
    a2 = VstSys.measuredSpectra[2, :]
    b2 = VstSys.measuredSpectra[3, :]

    pout = (np.abs(b2)**2) / (2*Z0)

    pin = (np.abs(a1)**2) / (2*Z0)

    pin_spectrum_db = 10*np.log10(pin) + 30
    pout_spectrum_db = 10*np.log10(pout) + 30

    pout_db = 10*np.log10(np.sum(pout)) + 30
    pin_db = 10*np.log10(np.sum(pin)) + 30
    gain_db = pout_db - pin_db

    # acpr = acpr_calculator.calc_acpr(pout)
    linearity_low, linearity_high, _ = linearity_calculator.calculate(pout)

    pin_t, pout_t, t = calc_td_powers(a1, np.zeros_like(a1), np.zeros_like(b2), b2, freqs)
    gain_cplx = pout_t / pin_t

    pin_t_db = 10*np.log10(np.abs(pin_t)) + 30


    am_am = 10*np.log10(np.abs(gain_cplx))
    am_pm = np.angle(gain_cplx, deg=True)


    # compute PAE, Pdc, Pout, Pin, Gain, IMD3/ACPR, AM_AM, AM_PM


    data = {
        "Pout_db": pout_db,
        "Pin_db": pin_db,
        "Gain_db": gain_db,
        "i_bias": i_bias,
        "i_main": i_main,
        "linearity_high_dbc": acpr_high,
        "linearity_low_dbc": acpr_low,
        "gain_mag_t": am_am,
        "gain_ang_t": am_pm,
        "pin_t_db": pin_t_db,
        "pin_spectrum_db": pin_spectrum_db,
        "pout_spectrum_db": pout_spectrum_db,
        "pa_td_waveform": rf_td_wavefrom,
        "dsm_td_waveform": dsm_td_waveform,
    }
    return data

output_dims = {
        "Pout_db": [],
        "Pin_db": [],
        "Gain_db": [],
        "i_bias": [],
        "i_main": [],
        "linearity_high_dbc": [],
        "linearity_low_dbc": [],
        "gain_mag_t": [("vst_time", t_vst)],
        "gain_ang_t": [("vst_time", t_vst)],
        "pin_t_db": [("vst_time", t_vst)],
        "pin_spectrum_db": [("frequency", freqs)],
        "pout_spectrum_db": [("frequency", freqs)],
        "pa_td_waveform": [("time", t_scope)],
        "dsm_td_waveform": [("time", t_scope)],

    }



ds = sweep_to_xarray_from_func(sweep, measure, output_dims=output_dims)

source_tuner.Source.off()
print(ds)
if IS_TWO_TONE:
    ds.attrs["linearity_metric"] = "IMD3"
else:
    ds.attrs["linearity_metric"] = "ACPR"
ds.attrs["signal"] = "CW"
ds.attrs["voltage_level"] = voltage_lvl
ds.attrs["dynamic_supply"] = DSM
ds.to_netcdf(f"two_tone_static_{voltage_lvl}v.h5", engine="h5netcdf", invalid_netcdf=True)
print(f"pout_db: {ds.Pout_db.values}")
print(f"pin_db: {ds.Pin_db.values}")
print(f"gain_db: {ds.Gain_db.values}")

plt.ioff()
fig, ax = plt.subplots()
ax.scatter(ds.pin_t_db.values[-1], ds.gain_mag_t.values[-1], label="am-am")
plt.legend()
plt.ylim((0,20))
plt.xlim(-100,100)
plt.show()

fig, ax = plt.subplots()
ax.scatter(ds.pin_t_db.values[-1], ds.gain_ang_t.values[-1], label="am-pm")
plt.legend()
plt.ylim((-180,180))
plt.xlim(-100,100)
plt.show()

fig, ax = plt.subplots()
ax.plot(ds.frequency.values, ds.pin_spectrum_db.values[-1], label="pin")
ax.plot(ds.frequency.values, ds.pout_spectrum_db.values[-1], label="pout")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ds.time.values, ds.pa_td_waveform.values[-1], label="pa")
# ax.plot(ds.time.values, ds.dsm_td_waveform.values[-1], label="dsm")
plt.legend()
plt.show()

print("DONE")
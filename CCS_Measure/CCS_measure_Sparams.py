#!/usr/bin/env python
# coding: utf-8

"""
Reyes Lucero 2024
╔═════╗  ╔═╗╔══╗  ╔═╗
║ ╔═╗ ║  ║ ║║ ╔╝  ║ ║
║ ╚═╝ ║  ║ ╚╝═╝   ║ ║
║ ╔╗ ╔╝  ║ ╔╗ ║   ║ ║ ╔═╗
║ ║║ ╚╗  ║ ║║ ╚╗  ║ ╚═╝ ║
╚═╝╚══╝  ╚═╝╚══╝  ╚═════╝
# A script for measuring DUT large signal performance across, excitation frequency, Vdd, and Pin
# This code does many things which could blow up your DUT if you are careless
# Please verify that all constants are appropriate for what you are trying to do.
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
from lib.VST_Measurement_System import MeasurementSystem
import skrf
from pathlib import Path


def dB(x):
    return 20 * np.log10(np.abs(x))


def save_sparams(sparams, freqs, fname):
    s_params = skrf.Network(f=freqs, f_unit="Hz", s=sparams, name="S-Params from CCS")
    f_path = Path(r"C:\Users\rfUser\Desktop")
    f_name = Path(f"{fname}.s2p")
    f = Path.joinpath(f_path, f_name)
    s_params.write_touchstone(f)


# ccsURL = "tcp://127.0.0.1:7531/CCS.rem"
ccsURL = "tcp://128.138.189.140:7531/CCS.rem"

# transistor bias levels and current limits
# make sure these are right if you care about your DUT
# dc_ports = {"Vgg_driver": 0, "MC_driver": 1, "Vgg_DUT": 2, "Vdd_driver_12w": 3, "DUT_drain": 4, ""}
GATE_SOURCE_IDX = 3
DRAIN_SOURCE_IDX = 0

DRAIN_IDX_LCL = 0
GATE_IDX_LCL = 2
# this is dumb, but it's the local index of the drain/gate virtual port on its analyzer

# GATE_ANALYZER_IDX = 1
# DRAIN_ANALYZER_IDX = 0
GATE_ANALYZER_IDX = 0
DRAIN_ANALYZER_IDX = 2
# these are indices for which output of the system supply is what on the
# DUT (zero based)


QUIESCENT_GATE = 2.1e-3  # A
QUIESCENT_DRAIN = 66e-3  # A
# QUIESCENT_DRAIN = 100e-3  # A
GATE_PINCH_OFF = -3.2
GATE_BIAS_INIT = -2.7  # V
GATE_CURRENT_LIMIT = 15e-3

GATE_MIN_MAX_V = [-4, -2.2]  # volts
DRAIN_MIN_MAX_V = [10, 28]  # volts
# GATE_BIAS_ACT = GATE_PINCH_OFF  # V
# after iteratively setting bias, we will store the bias point here.

DRAIN_BIAS_INIT = 28  # V
# DRAIN_MAX_CURRENT = 0.75  # A
DRAIN_MAX_CURRENT = 1.5  # A

DRAIN_OSC_THRESHOLD = 100e-3
# current value at which it's definitely oscillating.

GATE_STEP = 1e-2
# voltage steps to take when attempting to set the drain bias.
# If this is too small, the dc supply might not make any change.


# drain current limits for full sweep
# data sheet says 0.75A for CGH40006P
# FYI on the system supplies with two ranges
# setting the current limits too high will for the range will get it "stuck" in the lower voltage range

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

sparams, freqs = VST_Sys.measure_s_params(power_lvl=0)
# freqs = VST_Sys.freqs

# outlier_loc = np.where(dB(sparams[:, 0, 0]) == np.max(dB(sparams[:, 0, 0])))[0]
# print(f"Outlier Location: {outlier_loc}")
# print(f"nssb: {mod_range}")

''' Plot results '''
plt.figure('refl-abs')
plt.suptitle('refl-mag')
plt.plot(freqs / 1e9, dB(sparams[:, 0, 0]), label='|S11|')
plt.plot(freqs / 1e9, dB(sparams[:, 1, 1]), label='|S22|')
plt.xlabel('Frequency (GHz)')
plt.ylabel('|Sxy| (dB)')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.xlim((2.58, 2.62))
# plt.ylim((-40, 0))
plt.show()
# plt.savefig("thru_s21_refl_abs_-10dB_input", dpi=400)
plt.figure('transmission-abs')
plt.suptitle('transmission-mag')
plt.plot(freqs / 1e9, dB(sparams[:, 1, 0]), label='|S21|')
plt.plot(freqs / 1e9, dB(sparams[:, 0, 1]), label='|S12|')
plt.xlabel('Frequency (GHz)')
plt.ylabel('|Sxy| (dB)')
plt.legend()
# plt.xlim((2.55, 2.65))
# plt.ylim((-2, 2))
plt.grid()

plt.tight_layout()
plt.show()
# plt.savefig("thru_s21_transmission_abs_-10dB_input", dpi=400)
plt.figure('refl-phase')
plt.suptitle('reflection-phase')
plt.plot(freqs / 1e9, (np.angle(sparams[:, 0, 0], deg=True)), label='S11 (deg)')
plt.plot(freqs / 1e9, (np.angle(sparams[:, 1, 1], deg=True)), label='S22 (deg)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('angle(Sxy) (deg)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# plt.savefig("thru_s21_refl_phase_-10dB_input", dpi=400)

plt.figure('transmission-phase')
plt.suptitle('transmission-phase')
plt.plot(freqs / 1e9, (np.angle(sparams[:, 1, 0], deg=True)), label='S21 (deg)')
plt.plot(freqs / 1e9, (np.angle(sparams[:, 0, 1], deg=True)), label='S12 (Deg)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('angle(Sxy) (deg)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# save_sparams(sparams,freqs, r"C:\Users\relu4863\UCB-O365\Paul Flaten - WALP Paper\Data\k_taper")
# plt.savefig("thru_s21_transmission_phase_-10dB_input", dpi=400)
# print(f"Modulation Frequency: {rfAnalyzer.ModulationFrequency/1e6}MHz")


# todo plot am-am and am-pm in time

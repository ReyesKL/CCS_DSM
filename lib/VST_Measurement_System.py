#!/usr/bin/env python
# coding: utf-8

"""
Reyes Lucero 2023
╔═════╗  ╔═╗╔══╗  ╔═╗
║ ╔═╗ ║  ║ ║║ ╔╝  ║ ║
║ ╚═╝ ║  ║ ╚╝═╝   ║ ║
║ ╔╗ ╔╝  ║ ╔╗ ║   ║ ║ ╔═╗
║ ║║ ╚╗  ║ ║║ ╚╗  ║ ╚═╝ ║
╚═╝╚══╝  ╚═╝╚══╝  ╚═════╝
# VST System Measurement Wrapper Class
"""

# ## Importing required modules
import time
import clr

from lib.load_dll_libs import *
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import warnings

#test libs from paul
import lib.active_tuner_grid_lib as atg
#load the libraries
from lib.VST_Active_Tuner import VST_Tuner_Source, VST_Tuner_Receiver, VST_Active_Tuner, LinkedGammaDebugPlotter
from lib.emulated_network_lib import EmulatedNetwork
import lib.multitone_signal_lib as mts
from typing import Union
import lib.vst_util_lib as util
import os

#fix for duplicate KMP library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MeasurementSystem:
    @staticmethod
    def dB(x):
        return 20 * np.log10(np.abs(x))

    @staticmethod
    def dBm(x):
        return 20 * np.log10(np.abs(x)) + 10

    @staticmethod
    def polar_to_cplx(mag, ang, deg=True):
        if deg:
            ang = ang * (np.pi / 180)
        re = mag * np.cos(ang)
        im = mag * np.sin(ang)
        return re + 1j * im

    @staticmethod
    def ExtremeDoubleComplexToComlex(doubleComplex):
        return complex(doubleComplex.Re, doubleComplex.Im)


    def __init__(self, ccs_url, gate_src_idx, drain_src_idx, gate_idx_lcl, drain_idx_lcl, gate_analyzer_idx,
                 drain_analyzer_idx, quiescent_gate, quiescent_drain, gate_current_limit, drain_current_limit_init,
                 drain_current_limit_main, gate_min_max_v, drain_min_max_v, gate_pinch_off, gate_init_v, drain_init_v,
                 log, Z0=50, gate_v_step=0.01, settling_time=0.1):
        # todo maybe create a dc class so that can be be passed as one (optional) argument

        self.aligner_phase = 0.
        self.aligner_delay = 0.
        self.source2 = None
        self.source1 = None

        self.source1_signal = None
        self.source2_signal = None

        self.drain_src = None
        self.gate_src = None
        self.dc_analyzers_relevant = None
        self.rfAnalyzer = None
        self.drainAnalyzer = None
        self.gateAnalyzer = None
        self.aligner = None

        self.outputQuantities = [[0, 'af'], [0, 'bf'], [1, 'af'], [1, 'bf']]
        self.outputQuantitiesTime = [[0, 'at'], [0, 'bt'], [1, 'at'], [1, 'bt']]
        self.ports = [(0, Quantity.A), (0, Quantity.B), (1, Quantity.A), (1, Quantity.B)]

        self.ccsServer = IRemoteServer(ClientServices.OpenConnection(ccs_url))

        self.log = log

        self.init_measurement_system(gate_src_idx, drain_src_idx, gate_analyzer_idx, drain_analyzer_idx)

        # self.filename = f_name

        self.local_gate_index = gate_idx_lcl
        self.local_drain_index = drain_idx_lcl

        self.drain_osc_threshold = drain_current_limit_init

        self.rangeAndComplianceGate = DCGabaritLimits(gate_min_max_v, [gate_current_limit] * len(gate_min_max_v), 0)

        self.rangeAndComplianceDrainInit = DCGabaritLimits(drain_min_max_v,
                                                           [drain_current_limit_init] * len(drain_min_max_v), 0)
        # drain current limits for initial bias up
        self.rangeAndComplianceDrainFull = DCGabaritLimits(drain_min_max_v,
                                                           [drain_current_limit_main] * len(drain_min_max_v), 0)

        self.quiescent_gate = quiescent_gate
        self.quiescent_drain = quiescent_drain

        self.pinch_off_v = gate_pinch_off
        self.gate_init_v = gate_init_v
        self.drain_init_v = drain_init_v

        # self.drain_osc_threshold = drain_osc_threshold
        # self.meas_attrs = meas_attrs
        self.Z0 = Z0
        self.gate_v_step = gate_v_step
        self.settling_time = settling_time

        self.num_ssb_tones = int(self.rfAnalyzer.NrOfSSBModulationTones)
        self.mod_freq_indices = np.arange(-self.num_ssb_tones, self.num_ssb_tones + 1, 1)
        # todo these should be moved into getters

        self.ranges_range = self.rfAnalyzer.AvailableRangesAtPort(*self.ports[0])
        self.rfAnalyzer_ranges = np.asarray([val.Min for val in list(self.ranges_range)])
        self.rfAnalyzer_ranges = self.rfAnalyzer_ranges[:-4]
        self.measuredSpectra = np.empty((len(self.outputQuantities),
                                         2 * self.rfAnalyzer.NrOfSSBModulationTones + 1),
                                        dtype=complex)

        self.rfAnalyzer.Measure()  # just to get the freqs
        self.get_rf_data()

        self.freqs = np.array(list(map(float, self.rfAnalyzer.VirtualPorts[1]["af"].X)))
        self.grid_spacing = self.freqs[1] - self.freqs[0]

        self.rng = np.random.default_rng(seed=42)


        #tuners
        self.gamma_in_0             = None
        self.gamma_out_0            = None 
        self.gamma_l_0              = None
        self.gamma_s_0              = None
        self.reference_tuner        = None
        self.tuners                 = None
        self.__tuners_available     = False
        self.__tuners_setup         = False
        self.__source_grids         = []
        self.excitation_grids       = []
        self.tuner_grids            = []
        self.__set_source_gamma     = None
        self.__set_load_gamma       = None
        self.__source_signal        = None

        #emulated networks
        self.__source_net           = None
        self.__load_net             = None

    def init_aligner(self):
        rel_ph = np.array(list(map(float, self.source1.RelativeMultiTones.RelativePhases)))
        rel_amp = np.array(list(map(float, self.source1.RelativeMultiTones.RelativeAmplitudes)))
        rel_amp = 10 ** (rel_amp / 10)
        arr = self.polar_to_cplx(rel_amp, rel_ph, deg=True)
        # i think this needs to be the size of the measurement grid
        # so we need to put zeroes
        if rel_amp.size != self.measuredSpectra[0].size:
            full_arr = np.zeros_like(self.measuredSpectra[0])
            inxs = self.map_signal_to_measurement_grid(self.source1_signal) + self.num_ssb_tones
            full_arr[inxs] = arr
        else:
            full_arr = arr
        dotnet_array = System.Array[DoubleComplex]([DoubleComplex(item.real, item.imag) for item in full_arr])

        self.aligner = AlignerForPeriodicModulatedSignals(System.Double(self.grid_spacing), dotnet_array,
                                                          AlignerForPeriodicModulatedSignals.OptimizationMethod.NI_Correlation_2DMinimization)




    def init_measurement_system(self, gate_src_idx, drain_src_idx, gate_analyzer_idx, drain_analyzer_idx):
        # init ccs
        self.log.info(
            "Load Validation - CCS Install directory: {}".format(self.ccsServer.get_ICEInstallationDirectory()))

        # ## Get CCS Workspace Information

        activeSetupNames = self.ccsServer.get_SetupNames()
        self.log.info("Active CCS Workspace Setups Names: ")
        for name in activeSetupNames:
            self.log.info("{}".format(name))

        activeSchematics = self.ccsServer.get_ActiveSchematicNames()
        self.log.info("Active CCS Workspace Schematics: ")
        for name in activeSchematics:
            self.log.info("{}".format(name))

        # ## Get CCS Workspace Schematics
        analyzers = [s for s in self.ccsServer.ActiveSchematics if isinstance(s, RFAnalyzer)]
        dc_analyzers = [s for s in self.ccsServer.ActiveSchematics if isinstance(s, DCAnalyzer)]
        self.gateAnalyzer = dc_analyzers[gate_analyzer_idx]
        self.drainAnalyzer = dc_analyzers[drain_analyzer_idx]
        self.rfAnalyzer = analyzers[0]
        #
        self.dc_analyzers_relevant = [self.gateAnalyzer, self.drainAnalyzer]

        sources = [s for s in self.ccsServer.ActiveSchematics if isinstance(s, RFSource) or isinstance(s, RFSource_VST)]
        dc_sources = [s for s in self.ccsServer.ActiveSchematics if isinstance(s, DCSource)]

        self.gate_src = dc_sources[gate_src_idx]
        self.drain_src = dc_sources[drain_src_idx]
        # todo this could get a little scary
        #  no guarantee which is gate and which is drain.
        #  Marc references by their names which are descriptive in the CCS workspace.
        #  This is a much safer solution.

        self.source1 = sources[0]
        self.source2 = sources[2]

    def unbaias(self):
        try:
            self.drain_src.OutputEnabled = False
        except:
            self.log.error("Could not turn off drain supply! Leaving gate on")
            return
        time.sleep(0.5)
        self.gate_src.OutputEnabled = False

        self.drain_src.SetUserDefinedLimits(DCSourceMode.Voltage, self.rangeAndComplianceDrainInit)
        # reset the current limits so there's no nasty surprise

    def turn_off_all(self):
        self.log.info("Exiting...  Turning off RF and DC")
        try:
            self.source1.OutputEnabled = False
        except:
            self.log.warning("Could not turn of source 1, this could be very dangerous!")
        try:
            self.source2.OutputEnabled = False
        except:
            self.log.warning("Could not turn of source 2, this could be very dangerous!")

        self.unbaias()

    def check_gate(self, desired_voltage=None):
        if desired_voltage is None:
            desired_voltage = self.gate_init_v
        self.measure_dc()
        i_gate = list(map(self.ExtremeDoubleComplexToComlex,
                          self.gateAnalyzer.VirtualPorts[self.local_gate_index]["if"].Y))[0].real
        v_gate = list(map(self.ExtremeDoubleComplexToComlex,
                          self.gateAnalyzer.VirtualPorts[self.local_gate_index]["vf"].Y))[0].real

        self.log.info(f"Vgg={v_gate:.3f}V , Igg={i_gate * 1e3:.3f}mA")

        if np.abs(i_gate) >= self.quiescent_gate:
            self.log.error(f"Gate is drawing more than {self.quiescent_gate * 1e3}mA!")
            raise ValueError(f"Gate is drawing more than {self.quiescent_gate * 1e3}mA!")

        if not np.isclose(v_gate, desired_voltage, rtol=3e-2):
            self.log.error(f"Gate voltage was not properly applied!")
            raise ValueError("Gate voltage was not properly applied!")

        return v_gate, i_gate

    def check_drain(self, desired_voltage=None):
        # todo this causes unnecessary issues if quiescent current changes
        if desired_voltage is None:
            desired_voltage = self.drain_init_v
        self.measure_dc()
        i_drain = list(map(self.ExtremeDoubleComplexToComlex,
                           self.drainAnalyzer.VirtualPorts[self.local_drain_index]["if"].Y))[0].real
        v_drain = list(map(self.ExtremeDoubleComplexToComlex,
                           self.drainAnalyzer.VirtualPorts[self.local_drain_index]["vf"].Y))[0].real
        self.log.info(f"Vdd={v_drain:.3f}V , Idd={i_drain * 1e3:.3f}mA")

        if np.abs(i_drain) >= self.drain_osc_threshold:
            self.log.error(f"Drain is drawing more than {self.drain_osc_threshold * 1e3}mA!")
            raise ValueError(f"Drain is drawing more than {self.drain_osc_threshold * 1e3}mA!")

        if not np.isclose(v_drain, desired_voltage, rtol=1e-2):
            self.log.error("Drain voltage was not properly applied!")
            raise ValueError("Drain voltage was not properly applied!")

        return v_drain, i_drain

    def iteratively_set_gate(self, starting_voltage=None):
        if starting_voltage is None:
            starting_voltage = self.gate_init_v
        gate_voltage = starting_voltage
        assert self.gate_src.OutputEnabled
        assert self.drain_src.OutputEnabled
        self.drain_src.OutputLevel = self.drain_init_v
        _, i_drain = self.check_drain()
        prev_up = False
        while not np.isclose(i_drain, self.quiescent_drain, rtol=5e-2):

            if i_drain > self.quiescent_drain:
                gate_voltage -= self.gate_v_step
                prev_up = True
            else:
                if prev_up:
                    self.log.warning("Cannot converge to desired drain current.")
                    # raise ValueError
                    break  # this is a touch hacky

                gate_voltage += self.gate_v_step

            self.gate_src.OutputLevel = gate_voltage
            v_gate, _ = self.check_gate(gate_voltage)
            self.log.info(f"Set gate to {v_gate}V")
            time.sleep(2)
            _, i_drain = self.check_drain()

        self.log.info("Auto-bias Complete :)")
        return gate_voltage

    def pinch_off(self):
        self.gate_src.OutputLevel = self.pinch_off_v
        v_gate, _ = self.check_gate(gate_voltage)
        assert np.isclose(self.pinch_off_v, v_gate)
        self.log.info("Pinched off DUT")

    def squash_osc(self, desired_vdd, prev_vgg):
        self.log.warning("Squashing oscillators not implemented yet!")
        self.source1.OutputEnabled = False
        self.drain_src.OutputEnabled = False
        time.sleep(self.settling_time)

        self.drain_src.OutputEnabled = True
        time.sleep(self.settling_time)
        self.source1.OutputEnabled = True

    def directly_set_gate(self, gate_voltage):
        self.gate_src.OutputLevel = gate_voltage
        # iteratively setting is much safer.

    def setup_biasing(self):
        self.log.info("Beginning bias up procedure")
        self.drain_src.Reset()
        self.gate_src.Reset()
        self.drain_src.OutputEnabled = False
        # Safer to explicitly turn this off before attempting to bias,

        # voltage range, current limits, interpolation between two points - see GUI
        self.drain_src.SetUserDefinedLimits(DCSourceMode.Voltage, self.rangeAndComplianceDrainInit)

        self.gate_src.Polarity = DCPolarity.Reversed
        self.gate_src.SetUserDefinedLimits(DCSourceMode.Voltage, self.rangeAndComplianceGate)

        self.gate_src.OutputLevel = self.gate_init_v
        self.gate_src.OutputEnabled = True
        self.log.info("Gate On:")
        self.check_gate()
        self.log.info("Gate Checks Passed")

        time.sleep(2)
        self.drain_src.OutputLevel = self.drain_init_v
        self.drain_src.OutputEnabled = True
        self.log.info("Drain On:")

        _, i_drain = self.check_drain()
        self.log.info("Drain Checks Passed")

        set_gate_voltage = self.iteratively_set_gate()

        self.drain_src.SetUserDefinedLimits(DCSourceMode.Voltage, self.rangeAndComplianceDrainFull)
        self.log.info("Current limits raised to full measurement specs.")

        return True, set_gate_voltage
        # what the hek are we returning true for

    def get_rf_data(self):
        for i_quantity, outputQuantity in enumerate(self.outputQuantities):
            wv = \
                np.array(list(
                    map(self.ExtremeDoubleComplexToComlex,
                        self.rfAnalyzer.VirtualPorts[outputQuantity[0]][outputQuantity[1]].Y)))

            self.measuredSpectra[i_quantity, :] = wv

        return self.measuredSpectra

    def measure_dc(self):
        for dc_analyzer in self.dc_analyzers_relevant:
            dc_analyzer.Measure()
        return self.get_dc_data()

    def get_dc_data(self):
        v_gate = \
            list(
                map(self.ExtremeDoubleComplexToComlex, self.gateAnalyzer.VirtualPorts[self.local_gate_index]['vf'].Y))[
                0].real
        i_gate = \
            list(
                map(self.ExtremeDoubleComplexToComlex, self.gateAnalyzer.VirtualPorts[self.local_gate_index]["if"].Y))[
                0].real
        v_drain = \
            list(map(self.ExtremeDoubleComplexToComlex,
                     self.drainAnalyzer.VirtualPorts[self.local_drain_index]["vf"].Y))[
                0].real
        i_drain = \
            list(map(self.ExtremeDoubleComplexToComlex,
                     self.drainAnalyzer.VirtualPorts[self.local_drain_index]["if"].Y))[
                0].real
        return v_gate, i_gate, v_drain, i_drain

    def warm_up_dut(self, warm_up_pow_level):
        self.source1.OutputLevel = warm_up_pow_level
        self.source1.PlayMultitone((0, 0), (0, -100), (0, 0))
        # use a single tone excitation for warming up the DUT
        self.source1.OutputEnabled = True
        time.sleep(5)
        # this time is arbitrarily chosen.
        # There may be a way to determine the thermal time constant of the DUT
        # this is a little bit above my pay-grade
        self.source1.OutputEnabled = False
        self.log.info("DUT warmed up")


    def display_ranges(self):
        current_port_ranges = [self.rfAnalyzer.GetRangeAtPort(*port, self.ranges_range[0])[0] for port in self.ports]
        current_port_ranges = self.dBm(current_port_ranges)
        self.log.info(f"Set ranges to {current_port_ranges[0]:.1f}dBm A1, {current_port_ranges[1]:.1f}dBm B1, "
                      f"{current_port_ranges[2]:.1f}dBm A2, {current_port_ranges[3]:.1f}dBm B2")

    def get_available_port_ranges(self):  # , rf_analyzer, receivers):
        ranges = []
        for i_receiver, receiver in enumerate(self.ports):
            ranges.append([])
            available_ranges = self.rfAnalyzer.AvailableRangesAtPort(receiver[0], receiver[1])
            for i_range, available_range in enumerate(available_ranges):
                ranges[i_receiver].append(available_range.Min)
        return ranges

    def set_maximum_ranges(self):  # rf_analyzer, receivers):
        for receiver in self.ports:
            self.rfAnalyzer.SetRangeAtPort(receiver[0], receiver[1], 0, Range(0.0))

    def get_actual_ranges(self):  # rf_analyzer, receivers):
        ranges = []
        for i_recevier, receiver in enumerate(self.ports):
            rang, sth = self.rfAnalyzer.GetRangeAtPort(receiver[0], receiver[1], Range(0.0))
            ranges.append(rang)
        return ranges

    def measure(self, n_avg=1):
        s = np.shape(self.measuredSpectra)
        # dat = np.empty((n_avg, *s), dtype=np.complex)
        dat = np.zeros_like(self.measuredSpectra, dtype=complex)
        for i in range(n_avg):
            # try:
            #     self.rfAnalyzer.Measure()  # take a measurement
            # except:
            #     return False
            # todo need to keep this wrapped and figure out how to manage
            self.rfAnalyzer.Measure()
            dat += self.get_rf_data()  # get a and b waves

        self.measuredSpectra = dat / n_avg
        # return True

    def measure_and_set_ranges(self, num_avg=1):
        self.set_maximum_ranges()  # Set ranges to max so we don't over-range
        self.rfAnalyzer.Measure()  # take a measurement
        prev_ranges = np.array([-100, -100, -100, -100])
        curr_ranges = np.array([0, 0, 0, 0])

        ranges_delta = np.abs(prev_ranges - curr_ranges)
        # I have seen an oscillation with trying to get prev and current ranges to be equal
        # where it swaps between one range index, and it's neighbor.
        # So in an attempt to break the loop we will force the ranging algorithm to stop if the individual ranges are
        # within one index of each other.
        # then we will take the higher of the two possible ranges, set that and be on our way.
        n_iterations = 0
        while not np.all(ranges_delta <= 1) and n_iterations < 3:
            n_iterations += 1
            prev_ranges[:] = curr_ranges.data

            a1_td = np.abs(np.array(list(map(float, self.rfAnalyzer.VirtualPorts[0]['at'].Y))))
            b1_td = np.abs(np.array(list(map(float, self.rfAnalyzer.VirtualPorts[0]['bt'].Y))))
            a2_td = np.abs(np.array(list(map(float, self.rfAnalyzer.VirtualPorts[1]['at'].Y))))
            b2_td = np.abs(np.array(list(map(float, self.rfAnalyzer.VirtualPorts[1]['bt'].Y))))
            port_peak_volts = np.array([np.max(a1_td), np.max(b1_td), np.max(a2_td), np.max(b2_td)])
            # find the peak voltages each receiver saw

            # if port_peak_volts[3] < port_peak_volts[0]:
            #     self.log.error("DUT has no gain!")
            #     raise ValueError
            # this is a poor-man's attempt to verify that the DUT has gain

            for i_prt, port in enumerate(self.ports):
                port_range = np.searchsorted(np.flip(self.rfAnalyzer_ranges), port_peak_volts[i_prt], side='left')
                port_range = len(self.rfAnalyzer_ranges) - port_range - 2
                # find the range index corresponding to the nearest larger range to the seen peak voltage
                curr_ranges[i_prt] = port_range

                # find the range index corresponding to the nearest larger range to the seen peak voltage
                self.rfAnalyzer.SetRangeAtPort(*port, port_range, self.ranges_range[0])

            ranges_delta = np.abs(prev_ranges - curr_ranges)
            # self.rfAnalyzer.Measure()  # measure again now that ranges are set
            self.measure(n_avg=num_avg)
        # if np.sum(ranges_delta > 0):  # then we were in disagreement about at least one range
        if np.sum(ranges_delta) > 0:  # then we were in disagreement about at least one range

            largest_ranges = np.minimum(curr_ranges, prev_ranges)
            for i_prt, port in enumerate(self.ports):
                self.rfAnalyzer.SetRangeAtPort(*port, largest_ranges[i_prt] - 1, self.ranges_range[0])

            # self.rfAnalyzer.Measure()
            self.measure(n_avg=num_avg)

        # self.measure(n_avg=num_avg)

    def load_signal(self, source, sig):
        sig.phases *= (180 / np.pi)
        if source is self.source1:
            self.source1_signal = sig
        elif source is self.source2:
            self.source2_signal = sig
        else:
            self.log.error("RF Source not recognized")
            self.turn_off_all()
        tone_list = self.map_signal_to_measurement_grid(sig)
        # rel_powers = sig.amplitudes
        rel_powers = sig.amplitudes - np.min(sig.amplitudes)

        rel_phases = sig.phases
        source.PlayMultitone((tone_list), rel_powers, (rel_phases))
        # args are tone_idx, rel_powers, phases

    def map_signal_to_measurement_grid(self, sig):
        measurement_tone_spacing = self.rfAnalyzer.ModulationFrequency
        excitation_tone_spacing = sig.get_tone_spacing()
        if excitation_tone_spacing == np.NAN:
            raise Warning("Signal Tone Spacing must be set")

        if sig.get_grid_slots is np.NAN:
            raise Warning("Signal Grid must be set")

        grid_decimation_factor = int(excitation_tone_spacing / measurement_tone_spacing)
        # this is dangerous, we should look for remainders instead
        # todo switch to remainder approach

        sig_grid_slots = sig.get_grid_slots()
        sig_grid_slots -= np.min(sig_grid_slots)
        # this is also potentially dangerous
        tone_list = grid_decimation_factor * sig_grid_slots
        tone_list_center_idx = int(np.mean(tone_list))
        # this assumes some symmetry about the signal. I think?
        # tone_list_center_val = tone_list[tone_list_center_idx]
        tone_list -= tone_list_center_idx
        return tone_list


    def measure_s_params(self, power_lvl=-10, phase=None):
        mod_range = self.num_ssb_tones

        # played signal settings
        mod_freq_index = np.arange(-mod_range, mod_range + 1, 1)
        power_spectrum_dB = np.zeros(mod_freq_index.shape)
        # phase_spectrum_deg = np.random.uniform(-180, 180, size=mod_freq_index.shape)
        
        #handle the phase argument
        if isinstance(phase, np.ndarray):
            phase_spectrum_deg = phase
        elif isinstance(phase, str) and phase.lower() == "random":
            phase_spectrum_deg = np.random.uniform(-180, 180, size=mod_freq_index.shape)
        elif (phase is None) or (isinstance(phase, str) and phase.lower() == "schroeder"):
            N = mod_freq_index.size
            k=np.arange(1,N+1,1)
            phase_spectrum = np.pi*np.floor((k**2)/(2*N))
            phase_spectrum = (phase_spectrum + np.pi) % (2 * np.pi) - np.pi
            phase_spectrum_deg = np.rad2deg(phase_spectrum)
        else:
            raise TypeError("Unrecognized argument for phase.")

        sources = [self.source1, self.source2]
        for source in sources:
            source.OutputEnabled = False
        measuredSpectra = np.empty((len(sources), len(self.outputQuantities), 2 * mod_range + 1),
                                   dtype=complex)
        for i_source, source in enumerate(sources):

            source.PlayMultitone(mod_freq_index, power_spectrum_dB, phase_spectrum_deg)
            source.OutputLevel = power_lvl
            source.ModulationEnabled = True
            source.OutputEnabled = True
            # set_range_at_level(rfAnalyzer, receivers, 5)  # dBm
            self.measure_and_set_ranges()
            freqs = self.freqs

            for i_quantity, outputQuantity in enumerate(self.outputQuantities):
                measuredSpectra[i_source, i_quantity, :] = \
                    np.array(list(
                        map(self.ExtremeDoubleComplexToComlex,
                            self.rfAnalyzer.VirtualPorts[outputQuantity[0]][outputQuantity[1]].Y)))

            # turn off the source
            source.OutputEnabled = False

        # Create S matrix
        sparams = np.zeros((freqs.size, 2, 2), dtype=complex)
        for i_freq, freq in enumerate(freqs):
            # create wave matrices for both directions of excitation
            a_matrix = np.array([[measuredSpectra[0, 0, i_freq], measuredSpectra[1, 0, i_freq]],
                                 [measuredSpectra[0, 2, i_freq], measuredSpectra[1, 2, i_freq]]])
            b_matrix = np.array([[measuredSpectra[0, 1, i_freq], measuredSpectra[1, 1, i_freq]],
                                 [measuredSpectra[0, 3, i_freq], measuredSpectra[1, 3, i_freq]]])
            # solve for the S parameters
            sparams[i_freq, :, :] = b_matrix @ np.linalg.inv(a_matrix)

        return sparams, freqs


    ## SOME NEW METHODS
    """
    New Static Methods
    """

    def init_alignerV2(self,to_source=None):
        """ Initialize the aligner to signal played on one of the sources"""
        
        #set the source and signal to align to 
        if to_source is None:
            source = self.source1
            sig = self.source1_signal
        elif to_source is self.source1: 
            source = self.source1
            sig = self.source1_signal
        elif to_source is self.source2: 
            source = self.source2
            sig = self.source2_signal
        else:
            raise TypeError("provided source must be None, source1, or source2")

        rel_ph = np.array(list(map(float, source.RelativeMultiTones.RelativePhases)))
        rel_amp = np.array(list(map(float, source.RelativeMultiTones.RelativeAmplitudes)))
        rel_amp = 10 ** (rel_amp / 10)
        arr = self.polar_to_cplx(rel_amp, rel_ph, deg=True)
        # i think this needs to be the size of the measurement grid
        # so we need to put zeroes
        if rel_amp.size != self.measuredSpectra[0].size:
            full_arr = np.zeros_like(self.measuredSpectra[0])
            inxs = self.map_signal_to_measurement_grid(sig) + self.num_ssb_tones
            full_arr[inxs] = arr
        else:
            full_arr = arr
        dotnet_array = System.Array[DoubleComplex]([DoubleComplex(item.real, item.imag) for item in full_arr])

        self.aligner = AlignerForPeriodicModulatedSignals(System.Double(self.grid_spacing), dotnet_array,
                                                          AlignerForPeriodicModulatedSignals.OptimizationMethod.NI_Correlation_2DMinimization)
        
    def alignV2(self, arr, using_stored_alignment=False, keep_alignment=True, with_signal=None):
        #convert the array to a dotnet array
        arr = System.Array[DoubleComplex]([DoubleComplex(item.real, item.imag) for item in arr])

        # #initialize the aligner if it isn't already aligned 
        # if self.aligner is None:
        #     #will automatically use source 1
        #     self.init_alignerV2()


        #Perform alignment
        if not using_stored_alignment:
            if with_signal is not None:
                #alignment signal also needs to be a dot net array
                with_signal = System.Array[DoubleComplex]([DoubleComplex(item.real, item.imag) for item in with_signal])
                #create a new aligner for this purpose (using the reference signal)
                aligner = AlignerForPeriodicModulatedSignals(System.Double(self.grid_spacing),
                                                             with_signal,
                                                             AlignerForPeriodicModulatedSignals.OptimizationMethod.NI_Correlation_2DMinimization)
                #use the provided signal for alignment
                (delay, phaseoffset), arr = aligner.Align(arr, [])

                #save the new aligner if none already exists
                if self.aligner is None:
                    #store the generated asligner
                    self.aligner = aligner
            else:
                #re-run the aligner
                (delay, phaseoffset), arr = self.aligner.Align(arr, [])

            #store the new alignment if requested
            if keep_alignment:
                #update the class aligner delay and phase offsets
                self.aligner_delay = delay
                self.aligner_phase = phaseoffset
        else: #use prior delay and phase offsets
            #use the aligner with prior alignment offsets
            arr = self.aligner.ApplyDelay(arr, self.aligner_delay, self.aligner_phase)

        #Re-cast the array to a numpy array and return
        arr = np.array(list(map(self.ExtremeDoubleComplexToComlex, arr)))
        return arr

    # def apply_delta(self, delta, source, signal_index=1):
    #     """
    #     Apply Signal Delta to the specified source
    #     """
    #     #pull in the tone list
    #     # tone_list = self.get_tone_list(signal_index=signal_index)
    #     if signal_index == 1:
    #         tone_list = self.map_signal_to_measurement_grid(self.source1_signal)
    #     else: #signal_index == 2
    #         tone_list = self.map_signal_to_measurement_grid(self.source2_signal)
    #     # shifted_tone_list = np.add(self.num_ssb_tones, tone_list)
    #
    #     # Get the current source amplitudes and phases
    #     rel_phases = np.array(list(map(float, source.RelativeMultiTones.RelativePhases)))
    #     rel_amps = np.array(list(map(float, source.RelativeMultiTones.RelativeAmplitudes)))
    #
    #     # for the estimated amplitude error of a1_p
    #     delta_amp = 20 * np.log10(np.mean(np.abs(delta)))
    #     rel_delta_amp_db = 20 * np.log10(np.abs(delta))
    #     rel_delta_amp_db = rel_delta_amp_db - delta_amp
    #     # rel_delta_amp_db = np.minimum(rel_delta_amp_db, rel_amp_clamp_val)
    #     # rel_delta_amp_db = np.maximum(rel_delta_amp_db, -rel_amp_clamp_val)
    #     # delta_amp = np.min((delta_amp, amp_clamp_val))
    #     # delta_amp = np.max((delta_amp, -amp_clamp_val))
    #     # rel_amps += damping_fac * rel_delta_amp_db
    #     rel_amps += rel_delta_amp_db
    #     rel_amps -= np.mean(rel_amps)
    #
    #     # for the phase error
    #     delta_phase = 1 * np.angle(delta, deg=True)
    #     # delta_phase = np.minimum(delta_phase, phase_clamp_val)
    #     # delta_phase = np.maximum(delta_phase, -phase_clamp_val)
    #
    #     # rel_phases += damping_fac * delta_phase
    #     rel_phases += delta_phase
    #     rel_phases = np.where(rel_phases > 360, rel_phases - 360, rel_phases)
    #     rel_phases = np.where(rel_phases < 0, rel_phases + 360, rel_phases)
    #
    #     #update the source
    #     # source.OutputLevel += damping_fac * delta_amp
    #     source.OutputLevel += delta_amp
    #     # update the signal
    #     source.PlayMultitone((tone_list), rel_amps, rel_phases)

    def apply_delta(self, delta, source, tone_list):
        """
        Apply Signal Delta to the specified source
        """
        # Get the current source amplitudes and phases
        rel_phases = np.array(list(map(float, source.RelativeMultiTones.RelativePhases)))
        rel_amps = np.array(list(map(float, source.RelativeMultiTones.RelativeAmplitudes)))

        # for the estimated amplitude error of a1_p
        delta_amp = 20 * np.log10(np.mean(np.abs(delta)))
        rel_delta_amp_db = 20 * np.log10(np.abs(delta))
        rel_delta_amp_db = rel_delta_amp_db - delta_amp

        rel_amps += rel_delta_amp_db
        # rel_amps -= np.mean(rel_amps)
        rel_amps -= np.max(rel_amps)
        # for the phase error
        delta_phase = 1 * np.angle(delta, deg=True)

        # rel_phases += damping_fac * delta_phase
        rel_phases += delta_phase
        rel_phases = np.where(rel_phases > 360, rel_phases - 360, rel_phases)
        rel_phases = np.where(rel_phases < 0, rel_phases + 360, rel_phases)

        #update the source
        # source.OutputLevel += damping_fac * delta_amp
        source.OutputLevel += delta_amp
        # update the signal
        source.PlayMultitone((tone_list), rel_amps, rel_phases)

    def add_tones_from_grid(self, source, tone_locations:Union[np.ndarray[bool], np.ndarray[int]], ref_grid:atg.Grid, new_vals:np.ndarray[complex]):
        """
        Summary: Add new tones to the played signal 
        TODO: Add additional machinery for updating the output level.
        NOTE: This does not update the output level of the source at this time. Therefore it is advisable to keep relative amplitude 
        levels low and let the VST determine the appropriate update on the next relative update iteration. 

        ARGUMENTS:
        source - the source object that this method will act on 
        tone_locations - an array of indexes or a mask marking the locations of the tones to be added
        ref_grid: this is the frequency grid of the calling object that tone_locations is valid on 
        new_vals: New relative, complex powers to use for the intial values of the signal. Complex values are used here to remove ambiguity in phase (rad or deg).
        """

        #get the index of the source
        if source is self.source1:
            src_idx = 0
        elif source is self.source2:
            src_idx = 1
        else:
            raise TypeError("Unrecognized argument for source")

        #calculate the number of tones
        if tone_locations.dtype is np.dtype("int"):
            num_tones = tone_locations.size
        elif tone_locations.dtype is np.dtype("bool"):
            num_tones = np.sum(tone_locations)
        else:
            raise TypeError("Wrong data type for tone locations.")
        
        #don't waste time on this if the number of tones is 0
        if num_tones == 0:
            return

        #make sure the new values are specified as complex data
        assert isinstance(new_vals, np.ndarray), "new values must be specified as a numpy array"
        assert new_vals.dtype is np.dtype("complex"), "new values must be provided in complex format"
        assert num_tones == new_vals.size, f"Number of tones, {num_tones}, must match the number of new values, {new_vals.size}"

        #now get the new amplitudes and phases
        new_amps = util.db(new_vals)
        new_phases = util.complex2deg(new_vals)

        #get the source grid 
        src_grid = self.__source_grids[src_idx]

        #get the associated indices of the source grid on the tuner grid
        src_tone_list = src_grid.cast_index(ref_grid, about_center=True)
        new_tones = src_tone_list[tone_locations]

        #generate the tone indices to add to the played signal 
        _, current_tone_list, current_amps, current_phases = util.get_vst_source_multitone_data(source)

        #now add on the new tones and 
        updated_tone_list = np.concatenate((current_tone_list, new_tones))
        updated_amps      = np.concatenate((current_amps, new_amps))
        updated_phases    = np.concatenate((current_phases, new_phases))

        #return the unique elements in sorted order
        unq_srtd_idxs = np.unique(updated_tone_list, return_index=True)[1]

        #update the arrays
        updated_tone_list = updated_tone_list[unq_srtd_idxs]
        updated_amps = updated_amps[unq_srtd_idxs]
        updated_phases = updated_phases[unq_srtd_idxs]

        #now push everything to the source
        source.PlayMultitone(updated_tone_list, updated_amps, updated_phases)

    def rem_tones_from_grid(self, source, tone_locations:Union[np.ndarray[int], np.ndarray[bool]], ref_grid:atg.Grid):
        """
        Summary: Add new tones to the played signal 
        TODO: Add additional machinery for updating the output level.
        NOTE: This does not update the output level of the source at this time. Therefore it is advisable to keep relative amplitude 
        levels low and let the VST determine the appropriate update on the next relative update iteration. 

        ARGUMENTS:
        source - the source object that this method will act on 
        tone_locations - an array of indexes or a mask marking the locations of the tones to be added
        ref_grid: this is the frequency grid of the calling object that tone_locations is valid on 
        """

        #get the index of the source
        if source is self.source1:
            src_idx = 0
        elif source is self.source2:
            src_idx = 1
        else:
            raise TypeError("Unrecognized argument for source")
        
        #get the source grid 
        src_grid = self.__source_grids[src_idx]

        #get the associated indices of the source grid on the tuner grid
        src_tone_list = src_grid.cast_index(ref_grid, about_center=True)
        tones_to_remove = src_tone_list[tone_locations]

        #pull in the source data
        _, current_tone_list, current_amps, current_phases = util.get_vst_source_multitone_data(source)

        #search for the matching tones in the source's current tone list
        #time for some matlab magic, let's see if pesky python can keep up...
        tones_to_remove=tones_to_remove[:,np.newaxis] 
        search_results = tones_to_remove==current_tone_list
        match_found = np.any(search_results, axis=0)

        #keep only values where the match wasn't found
        updated_tone_list = current_tone_list[np.logical_not(match_found)]
        updated_amps = current_amps[np.logical_not(match_found)]
        updated_phases = current_phases[np.logical_not(match_found)]

        #now push everything to the source
        source.PlayMultitone(updated_tone_list, updated_amps, updated_phases)

    def apply_delta_from_grid(self, delta, to_source, from_grid:atg.Grid, to_tones:Union[np.ndarray[bool],None]=None)->None:
        ## applies a delta from another grid and ignores other entries
        
        """
        This is an updated version of the apply_delta_from_grid method that adds functionality for masking out 
        specific tones. This is important as the tuner will be removing tones from consideration in an ad-hoc 
        manner. The effective functionality of the original method "apply_delta_from_grid" is retained if the 
        to_tones variable is left unset.

        It is implicitely assumed that to_tones is a boolean array the same size as the measurement grid. No checks 
        are performed to validate this.
        """

        #get the index of the source
        if to_source is self.source1:
            src_idx = 0
        elif to_source is self.source2:
            src_idx = 1
        else:
            raise TypeError("Unrecognized argument for to_source")
        
        #handle the default case of the to_tones argument
        if to_tones is None:
            #create a mask of all true values
            to_tones = np.full(from_grid.size, True)

        #build the indices that will be used 
        src_grid = self.__source_grids[src_idx]

        #generate the tone indices that will be set on the source grid
        src_tone_list = src_grid.cast_index(from_grid, about_center=True)

        #now pass the delta along to the source
        self.apply_delta(delta[to_tones], to_source, src_tone_list[to_tones])

    def apply_absolute_from_grid(self, newVal, to_source, from_grid:atg.Grid, using_tones:Union[np.ndarray,None]=None):
        #apply aboslute signal from grid

        #create the default mask if none has been provided
        if using_tones is None:
            #default value for using tones is an array of true values
            using_tones = np.full(from_grid.size, True)
        elif not (isinstance(using_tones, np.ndarray) and (using_tones.dtype is np.dtype("bool"))):
            #mask must be an np.ndarray[bool] to pass this check
            raise TypeError("Unrecognized type for argument using_tones. Must be an array of type numpy.ndarray[bool]")
        elif (not using_tones.size == from_grid.size):
            #make sure the tones are the same size as the grid
            raise ValueError(f"Tone mask size, {using_tones.size}, does not match grid size, {from_grid.size}")
            
        
        #get the source index to use
        if to_source is self.source1:
            src_idx = 0
        elif to_source is self.source2:
            src_idx = 1
        else:
            raise TypeError("Unrecognized argument for to_source")

        #build the indices that will be used 
        src_grid = self.__source_grids[src_idx]

        #generate the tone indices that will be set on the source grid
        src_tone_list = src_grid.cast_index(from_grid, about_center=True)
        src_tone_list = src_tone_list[using_tones]

        #now create a new signal from the tone values provided
        temp_sig = mts.MultitoneSignal(from_grid.size, from_grid, auto_level=False)
        
        #set the masked values to zero
        tone_vals = newVal
        tone_vals[np.logical_not(using_tones)] = 0.0

        #the tone values need to be loaded in as voltage wave values
        temp_sig.v0 = tone_vals

        #get the peak time-domain power in dbm, i.e. what the vst output level is set to
        peak_power = 10*np.log10(temp_sig.peak_power) + 30

        #finally, get the relative tone values (normalized in magnitude to the peak tone value)
        rel_tone_powers = temp_sig.p0_normalized
        #apply tone mask and convert to db 
        rel_tone_powers = 10*np.log10(rel_tone_powers[using_tones])

        #get the phases of each tone
        tone_phases     = np.rad2deg(np.angle(temp_sig.v0))
        tone_phases     = tone_phases[using_tones]

        #now update the VST
        if peak_power <= 30:
            #just for now to make sure I don't break anything limit the power level to 10dBm
            to_source.OutputLevel = peak_power
            to_source.PlayMultitone((src_tone_list), rel_tone_powers, (tone_phases))
        else:
            raise RuntimeError(f"Attempted to set VST output level to: {peak_power:.2f}dBm")

    def measureV2(self, set_ranges=True, n_avgs=1, perform_alignment=False, use_expected_source=False):
        """ 
        Perform measurements while also providing mechanism for aligning
        """
        
        #make sure that the number of averages is >= 1
        assert n_avgs >= 1, "Number of averages must be >= 1"

        #perform initial measurement
        if set_ranges:
            #run measure and set ranges
            self.measure_and_set_ranges()
        else: #set_ranges = False
            #just run measure
            self.measure()
        
        #generate the initial results
        a1 = (self.measuredSpectra[0].copy()) / n_avgs
        b1 = (self.measuredSpectra[1].copy()) / n_avgs
        a2 = (self.measuredSpectra[2].copy()) / n_avgs
        b2 = (self.measuredSpectra[3].copy()) / n_avgs

        #use a1 as the alignment signal
        a10 = self.measuredSpectra[0].copy()  

        for idx in range(1, n_avgs):
            #now perform a new measurement
            self.measure()
            
            #get the new signal
            a1_next = self.measuredSpectra[0]
            b1_next = self.measuredSpectra[1]
            a2_next = self.measuredSpectra[2]
            b2_next = self.measuredSpectra[3]

            #now align a1_next to a10
            a1_next = self.alignV2(a1_next, with_signal=a10)
            b1_next = self.alignV2(b1_next, using_stored_alignment=True)
            a2_next = self.alignV2(a2_next, using_stored_alignment=True)
            b2_next = self.alignV2(b2_next, using_stored_alignment=True)

            #now update the running average values
            a1 += a1_next / n_avgs
            b1 += b1_next / n_avgs
            a2 += a2_next / n_avgs
            b2 += b2_next / n_avgs
    
        #now update the measured spectra
        self.measuredSpectra[0, :] = a1
        self.measuredSpectra[1, :] = b1
        self.measuredSpectra[2, :] = a2
        self.measuredSpectra[3, :] = b2

        #if we want to perform alignment 
        if perform_alignment:
            #do not align 
            if self.reference_tuner is None:
                self.log.warning("Reference tuner not set! Nothing to align to.")
            elif not self.reference_tuner.is_initialized:
                raise RuntimeError("Reference tuner has not been initialized yet. Cannot perform alignment.")
            else:
                #get the reference signal
                if use_expected_source:
                    ref = self.reference_tuner.A0_expected
                else:
                    ref = self.reference_tuner.A0

                #get properties from the reference tuner
                gamma_s_0 = self.reference_tuner.gamma_0
                a = self.reference_tuner.Receiver.a
                b = self.reference_tuner.Receiver.b

                #now calculate the unaligned source signal
                a = self.measurement_grid.cast(a, self.reference_tuner.grid)
                b = self.measurement_grid.cast(b, self.reference_tuner.grid)
                a0 = (a - b*gamma_s_0) / (1 - gamma_s_0)
                
                #cast both back to the measurement grid (required for alignment)
                a0 = self.reference_tuner.grid.cast(a0, self.measurement_grid)
                ref = self.reference_tuner.grid.cast(ref, self.measurement_grid)

                #get the unaligned 
                self.alignV2(a0, with_signal=ref)

                #re-align and update the signals
                self.measuredSpectra[0, :] = self.alignV2(a1, using_stored_alignment=True)
                self.measuredSpectra[1, :] = self.alignV2(b1, using_stored_alignment=True)
                self.measuredSpectra[2, :] = self.alignV2(a2, using_stored_alignment=True)
                self.measuredSpectra[3, :] = self.alignV2(b2, using_stored_alignment=True)

    def measureV3(self, reference_tuner=None, set_ranges=True, n_avgs=1, perform_alignment=False, use_expected_source=False):
        """ 
        Perform measurements while also providing mechanism for aligning
        """
        
        #make sure that the number of averages is >= 1
        assert n_avgs >= 1, "Number of averages must be >= 1"

        #perform initial measurement
        if set_ranges:
            #run measure and set ranges
            self.measure_and_set_ranges()
        else: #set_ranges = False
            #just run measure
            self.measure()
        
        #generate the initial results
        a1 = (self.measuredSpectra[0].copy()) / n_avgs
        b1 = (self.measuredSpectra[1].copy()) / n_avgs
        a2 = (self.measuredSpectra[2].copy()) / n_avgs
        b2 = (self.measuredSpectra[3].copy()) / n_avgs

        #use a1 as the alignment signal
        a10 = self.measuredSpectra[0].copy()  

        for idx in range(1, n_avgs):
            #now perform a new measurement
            self.measure()
            
            #get the new signal
            a1_next = self.measuredSpectra[0]
            b1_next = self.measuredSpectra[1]
            a2_next = self.measuredSpectra[2]
            b2_next = self.measuredSpectra[3]

            #now align a1_next to a10
            a1_next = self.alignV2(a1_next, with_signal=a10)
            b1_next = self.alignV2(b1_next, using_stored_alignment=True)
            a2_next = self.alignV2(a2_next, using_stored_alignment=True)
            b2_next = self.alignV2(b2_next, using_stored_alignment=True)

            #now update the running average values
            a1 += a1_next / n_avgs
            b1 += b1_next / n_avgs
            a2 += a2_next / n_avgs
            b2 += b2_next / n_avgs
    
        #now update the measured spectra
        self.measuredSpectra[0, :] = a1
        self.measuredSpectra[1, :] = b1
        self.measuredSpectra[2, :] = a2
        self.measuredSpectra[3, :] = b2

        #if we want to perform alignment 
        if perform_alignment:
            #do not align 
            if reference_tuner is None:
                self.log.warning("Reference tuner not valid. Nothing to align to.")
            elif not reference_tuner.is_initialized:
                raise RuntimeError("Reference tuner has not been initialized yet. Cannot perform alignment.")
            else:
                #get the reference signal
                if use_expected_source:
                    ref = reference_tuner.A0_expected
                else:
                    ref = reference_tuner.A0

                #get properties from the reference tuner
                gamma_s_0 = reference_tuner.gamma_0
                a = reference_tuner.Receiver.a
                b = reference_tuner.Receiver.b

                #now calculate the unaligned source signal
                a = self.measurement_grid.cast(a, reference_tuner.grid)
                b = self.measurement_grid.cast(b, reference_tuner.grid)
                a0 = (a - b*gamma_s_0) / (1 - gamma_s_0)
                
                #cast both back to the measurement grid (required for alignment)
                a0 = reference_tuner.grid.cast(a0, self.measurement_grid)
                ref = reference_tuner.grid.cast(ref, self.measurement_grid)

                #get the unaligned 
                self.alignV2(a0, with_signal=ref)

                #re-align and update the signals
                self.measuredSpectra[0, :] = self.alignV2(a1, using_stored_alignment=True)
                self.measuredSpectra[1, :] = self.alignV2(b1, using_stored_alignment=True)
                self.measuredSpectra[2, :] = self.alignV2(a2, using_stored_alignment=True)
                self.measuredSpectra[3, :] = self.alignV2(b2, using_stored_alignment=True)
    #New Setup and Initialization Methods


    """
    Methods for Tuner Creation and Initialization
    Tuners are not automatically created and initialized during VST bench construction. To get the tuners 
    up and running the following steps must be taken:
        1) Tuners Creation
            - With the create_tuners method
            - Tuners now operate on a heirarchical frequency grid system (see below)
            - Tasks:
                * Initialize the tuner grids, i.e. self.__build_tuner_grids()
                * Set the reference for measurement alignment 
                * Construct the tuners
                * Internally mark the tuners as available
        2) Tuner Setup
            - With setup_tuners method
            - Initializes the tuners
            - Initializes the signal of the tuners 
            - Initializes each tuner's A0 with initial measurement (T0 will be the difference between S0 and A0)
    Grid structure:
        The tuner structure uses a heirarchical grid approach to frequency and index management. As of writing this 
        doucmentation there are four types of grids present with an optional additional grid not implemented yet:
            1) Measurement Grid
                This is the base grid of the class and contains all frequencies that the VST system is measuring at.
                All other grids exist as a subset of this grid, i.e. all frequency points of other grids can be found
                on this grid. 
            2) Source Grid
                This is a continuous, uniform grid of frequencies where the transmitter COULD transmit, but not necissarily 
                where it will transmit. This grid is specifically used to relate sub-grids with the tone index notation used
                by the transciever hardware. 
            3) Excitation Grid
                A grid that is not necissarily continuous or uniform where source excitations exist. 
            4) Tuner Grid
                A grid where the total tuner excitation, A0, exists and reflection excitation, T0, exists
            5) Signal Grid
                A grid where the tuner signal excitation, S0, exists and must be on the tuner grid excitation. 
            6) Static Excitation Grid
                A grid for handling static excitations that are not directly governed by the tuners. External signals, simple generators, 
                interference, etc. This is simply added as a catch-all in the heirarchy and isn't implemented at this time. 
            7) Sense Grid
                Not implemented yet, but a grid that can be used to sense and update the tuner with information 
                on a DUT's large-signal gamma_in. The plan is to use this for either tickle tones, or 
                the gamma_in measurement method derived during tuner abstraction.
        

        Grid Heirarchy:
            -->Measurement Grid
                | ## FOR THE SOURCE TUNER ##
                |--> __source_grid (@ self.__source_grids[0])
                |   |
                |   --->excitation_grid (@ self.excitation_grids[0])
                |       |
                |       |-->tuner_grid (source in-band signal)
                |       |   |
                |       |   --->tuner_signal_grid
                |       |
                |       --->static_excitation_grid (if you don't want a tuner)
                |       |
                |       --->sense_grid (for sensing large-signal gamma_in, not implemented yet but anticipated)
                |
                | ## FOR THE LOAD TUNER ##
                |--> __source_grid (@ self.__source_grids[1])
                    |
                    --->excitation_grid (@ self.excitation_grids[1])
                        |
                        |-->tuner_grid (load in-band signal)
                        |   |
                        |   --->tuner_signal_grid
                        |
                        --->static_excitation_grid
                        |
                        --->sense_grid (for sensing large-signal gamma_in, not implemented yet but anticipated)
                

    """

    def create_tuners(self, ref_channel_idx:int=1)->None:
        # Create tuners for this system 

        #build the tuner grids
        self.__build_tuner_grids()

        #initialize the tuner list
        self.tuners = []
        
        #setup the source objects
        source_objects = [self.source1, self.source2]

        tuner_names = ["Source", "Load"]

        #set up the first channel 
        for idx in range(0,2):
            #get the present channel index
            pres_channel = idx + 1

            #create the next receiver
            next_receiver = VST_Tuner_Receiver(self, pres_channel, self.measurement_grid, is_tuner_ref=(pres_channel==ref_channel_idx))
            next_source = VST_Tuner_Source(self, pres_channel, source_objects[idx], self.excitation_grids[idx])
            
            #create the next tuner 
            next_tuner = VST_Active_Tuner(tuner_names[idx], next_source, next_receiver, self.tuner_grids[idx])
            
            #set the current tuner as the reference tuner if the channel is the reference channel
            if pres_channel == ref_channel_idx:
                self.reference_tuner = next_tuner

            #append the tuner to the list
            self.tuners.append(next_tuner)

        #throw an error if the reference channel hasn't been set
        if self.reference_tuner is None:
            self.log.warning("Reference tuner not set! Nothing to align to.")
        
        #set tuners available to true
        self.__tuners_available = True
    
    def find_reflection_coefficients(self, source_1_power:float=0, source_2_power:float=0, use_grid:str="source", return_power_lvl:float=-80):
        """
        Finds the reflection coefficients of the dut and sources over the source grids.
        Note: At this point in time, this method uses the established source grids. This means
        that the tuners should be set up before running.

        TODO: This method should really be using the established tuner objects to determine the 
        associated reflection coefficients. 
        """
        #find the reflection coefficients of the dut over the entire measurement grid
        if not (self.__source_grids[0] == self.__source_grids[1]):
            raise NotImplementedError("Handling non-equivalent source grids has not been implemented yet.")

        #Make sure all sources are off
        self.source1.OutputEnabled = False
        self.source2.OutputEnabled = False
        
        #Makes sure modulation is enabled
        self.source1.ModulationEnabled = True
        self.source2.ModulationEnabled = True

        """
        Part 1: Find "small-signal" parameters with source 1 excitation
        """
        if use_grid=="source":
            target_grid = self.__source.grids[0]
        elif use_grid=="tuner":
            target_grid = self.tuners[0].grid
        elif use_grid=="excitation":
            target_grid = self.excitation_grids[0]
        else:
            raise ValueError("Specified grid must be source, tuner, or excitation")
        
        #get the source grid
        source_grid = self.__source_grids[0]
        
        #get the relative amplitudes and phases
        rel_amplitudes = np.full(target_grid.size, 0.0)
        rel_phases = np.random.uniform(-180, 180, target_grid.size)

        #get the indices of the source grid
        src_idxs = source_grid.cast_index(target_grid, about_center=True)
        
        #now load the signal into the source
        self.source1.PlayMultitone(src_idxs, rel_amplitudes, rel_phases)
        #set the output level to the first source
        self.source1.OutputLevel = source_1_power
        #turn on the first source 
        self.source1.OutputEnabled = True

        #make a measurement
        self.measure_and_set_ranges()
        #turn off the source
        self.source1.OutputEnabled = False
        self.source1.OutputLevel = return_power_lvl
        #get the a/b waves     
        a1 = self.measuredSpectra[0]
        b1 = self.measuredSpectra[1]
        a2 = self.measuredSpectra[2]
        b2 = self.measuredSpectra[3]
        #find the gammas
        gamma_in_0 = b1 / a1
        gamma_l_0  = a2 / b2

        
        """
        Part 2: Find "small-signal" parameters with source 2 excitation
        """
        if use_grid=="source":
            target_grid = self.__source.grids[1]
        elif use_grid=="tuner":
            target_grid = self.tuners[1].grid
        elif use_grid=="excitation":
            target_grid = self.excitation_grids[1]
        else:
            raise ValueError("Specified grid must be source, tuner, or excitation")
        
        #get the source grid
        source_grid = self.__source_grids[1]
        
        # source_grid = self.tuners[1].grid
        rel_amplitudes = np.full(target_grid.size, 0.0)
        rel_phases = np.random.uniform(-180, 180, target_grid.size)

        #get the indices of the source grid
        src_idxs = source_grid.cast_index(target_grid, about_center=True)
        
        #now load the signal into the source
        self.source2.PlayMultitone(src_idxs, rel_amplitudes, rel_phases)
        #set the output level to the first source
        self.source2.OutputLevel = source_2_power
        #turn on the first source 
        self.source2.OutputEnabled = True
        #make a measurement
        self.measure_and_set_ranges()
        #turn off the source
        self.source2.OutputEnabled = False
        self.source2.OutputLevel = return_power_lvl
        #get the a/b waves     
        a1 = self.measuredSpectra[0]
        b1 = self.measuredSpectra[1]
        a2 = self.measuredSpectra[2]
        b2 = self.measuredSpectra[3]
        #find the gammas
        gamma_out_0 = b2 / a2
        gamma_s_0   = a1 / b1
        

        """
        Part 3: Upload the reflection coefficients to the tuners
        """
        #for the source tuner
        source_tuner = self.tuners[0]
        source_tuner.gamma_0    = self.measurement_grid.cast(gamma_s_0, source_tuner.grid)
        source_tuner.gamma_in_0 = self.measurement_grid.cast(gamma_in_0, source_tuner.grid)

        #for the load tuner
        load_tuner = self.tuners[1]
        load_tuner.gamma_0      = self.measurement_grid.cast(gamma_l_0, load_tuner.grid)
        load_tuner.gamma_in_0   = self.measurement_grid.cast(gamma_out_0, load_tuner.grid)

        """
        Finally, save the tuner gammas to the VST object (for now)
        """
        self.gamma_in_0 = gamma_in_0
        self.gamma_out_0= gamma_out_0
        self.gamma_s_0  = gamma_s_0
        self.gamma_l_0  = gamma_l_0
    
    def setup_tuners(self, signal_power:float=0.001, with_signal=None, interpret_signal_phases_as_deg:bool=True):
        #Sets up the tuners as load and source-pull tuners

        #check if the tuners are setup
        if not self.__tuners_available:
            raise RuntimeError("Tuners are not available yet. Please run create_tuners first.")
        
        #turn both tuners on 
        self.source1.OutputEnabled=True
        self.source2.OutputEnabled=True

        #make an initial measurement here (no need to align)
        self.measureV2()
        a1 = self.measuredSpectra[0]
        b1 = self.measuredSpectra[1]
        a2 = self.measuredSpectra[2]
        b2 = self.measuredSpectra[3]

        #NOTE: For now we assume that the signal grid is the same as the excitation grid for both tuners
        # sig_grid = self.excitation_grids[0]

        """
        For the source pull tuner
            1) gamma_0 is gamma_s_0
            2) gamma_in_0 (assigned externally)
            3) S0 is the signal
            4) T0 is the tuner value
        """
        source_tuner = self.tuners[0]
        if source_tuner.gamma_0 is None:
            if not self.gamma_s_0 is None:
                source_tuner.gamma_0 = self.gamma_s_0
            else:
                raise RuntimeError("gamma_0 could not be set on source tuner.")
        
        if source_tuner.gamma_in_0 is None:
            if not self.gamma_in_0 is None:
                source_tuner.gamma_in_0 = self.gamma_in_0
            else:
                raise RuntimeError("gamma_in_0 could not be set on source tuner.")
            
        #now add the signal 
        if with_signal is None: 
            vst_sig = self.source1_signal
        else:
            vst_sig = with_signal 

        #get the source center and offset frequencies (directly from the source) for signal grid creation
        center_frequency = self.source1.Frequency
        offset_frequency = self.source1.MultiToneFrequency 

        #create the source signal
        source_signal = mts.MultitoneSignal.from_vst_signal(vst_sig, 
                                                            self.tuner_grids[0], 
                                                            power=signal_power,
                                                            phase_in_deg=interpret_signal_phases_as_deg,
                                                            source_grid_freq_step=offset_frequency, 
                                                            source_grid_center_freq=center_frequency)

        
        #get the reference signal to use for bootstrapping the tuner
        # idxs = self.get_tone_list(shift=True)
        # a10 = (a1[idxs] - b1[idxs]*source_tuner.gamma_0)/(1 - source_tuner.gamma_0)
        a1 = self.measurement_grid.cast(a1,source_tuner.grid)
        b1 = self.measurement_grid.cast(b1,source_tuner.grid)
        a10 = (a1 - b1*source_tuner.gamma_0)/(1 - source_tuner.gamma_0)

        #vst works in voltage not sqrt of power
        # source_tuner.initialize(A0=a10, S0=source_signal.v0)
        # source_tuner.initialize(A0=a10, S0=source_signal.a0)
        source_tuner.initialize(A0=a10, S0=source_signal) #have the tuner use the multitone signal as a reference
        self.__source_signal = source_signal
        #now set the signal for the source tuner
        #set the signal 



        """
        For the load pull tuner
            1) gamma_0 is gamma_l_0
            2) gamma_in_0 (assigned externally)
            3) S0 is zero
            4) T0 (leave alone)
        """
        load_tuner = self.tuners[1]
        if load_tuner.gamma_0 is None:
            if not self.gamma_l_0 is None:
                load_tuner.gamma_0 = self.gamma_l_0
            else:
                raise RuntimeError("gamma_0 could not be set on load tuner.")
        
        if load_tuner.gamma_in_0 is None:
            if not self.gamma_out_0 is None:
                load_tuner.gamma_in_0 = self.gamma_out_0
            else:
                raise RuntimeError("gamma_in_0 could not be set on load tuner.")
            
        #tuner 2 doesn't have a signal to use
        load_signal = mts.MultitoneSignal.null_signal(self.tuner_grids[1])

        #get the reference signal to use for bootstrapping the tuner
        # idxs = self.get_tone_list(shift=True)
        # a20 = (a2[idxs] - b2[idxs]*load_tuner.gamma_0)/(1 - load_tuner.gamma_0)
        a2 = self.measurement_grid.cast(a2,load_tuner.grid)
        b2 = self.measurement_grid.cast(b2,load_tuner.grid)
        a20 = (a2 - b2*load_tuner.gamma_0)/(1 - load_tuner.gamma_0)

        #vst works in voltage not sqrt of power (S0 + T0 = A0)
        # load_tuner.initialize(A0=a20, S0=load_signal.v0)
        load_tuner.initialize(A0=a20, S0=load_signal.a0)
        
        #initialize the set gamma values
        self.__set_source_gamma = np.full(source_tuner.grid.size, 0, dtype=np.dtype("complex"))
        self.__set_load_gamma = np.full(load_tuner.grid.size, 0, dtype=np.dtype("complex"))

        #record that the tuners are setup 
        self.__tuners_setup = True

        #return the source and load signals
        return source_signal, load_signal
    
    def __build_tuner_grids(self)->None:
        ## Build the grids for the tuners to use

        #first, build the measurement grid as the base grid for the tuners
        self.measurement_grid = self.__generate_base_grid()

        self.excitation_grids = []
        self.__source_grids = []
        #next, build the source excitation grids

        #for source 1...
        next_source_grid, next_excitation_grid = self.__generate_source_and_excitation_grids("1", self.source1) 
        self.__source_grids.append(next_source_grid)
        self.excitation_grids.append(next_excitation_grid)

        #for source 2...
        next_source_grid, next_excitation_grid = self.__generate_source_and_excitation_grids("2", self.source2) 
        self.__source_grids.append(next_source_grid)
        self.excitation_grids.append(next_excitation_grid)

        #finally, add the tuner responses 
        # TODO: For now these will be the same as the excitation grids
        self.tuner_grids = self.excitation_grids

    def __generate_base_grid(self)->atg.Grid:
        
        #get a list of all frequencies on the measurement grid
        freqs = self.freqs

        #build the grid generator(passing center as type None will use the mean frequency as the reference)
        grid_generator = atg.GridGenerator(atg.StaticGridSource(freqs, center=None))

        #build the base grid
        return atg.Grid.generate("Measurement", using=grid_generator)
    
    def __generate_source_and_excitation_grids(self, name:str, source:RFSource)->tuple[atg.Grid, atg.Grid]:
        #Updated version of source and excitation grid generation. In this version, the extense of the 
        #source grid is extended to the edges of the measurement grid. The excitation grid is still pulled
        #directly from the VST source.
        #Notes: 
        #   a) This assumes that the base grid is the measurement grid, and
        #   b) that the selected source excitation is set up
        #TODO: Add a check to make the source grid the same as the measurement grid if the step and centers are the same
        
        #get the source frequencies
        center_frequency = source.Frequency
        offset_frequency = source.MultiToneFrequency
        
        #some setup for building the source grid
        lf = self.freqs[0]  #lowest measurement frequency
        hf = self.freqs[-1] #highest measurement frequency
        fs = self.freqs[1] - self.freqs[0] #measurement frequency step
        fc = np.mean(self.freqs)

        #make sure that the offset frequency is an integer multiple of the measurement frequency step
        step_mult = (float(offset_frequency) / float(fs))
        err = np.abs(step_mult - np.rint(step_mult))
        assert err < 1e-6, "Source frequency step is not an integer multiple of the measurement grid step size"

        #determine the number of steps from the center frequency to the low end of the measurement grid
        lf_ns = np.floor((center_frequency - lf) / offset_frequency)

        #determine the nubmer of steps from the center frequency to the high end of the measurement grid
        hf_ns = np.floor((hf - center_frequency) / offset_frequency)

        #build the range of indices to multiply and offset by the center frequency
        src_range = np.arange(-lf_ns, hf_ns+1) #Note: +1 added to include the final point
        src_freqs = src_range*offset_frequency + center_frequency

        #pull the "indices" of the excitation tones on the grid
        net_idxs = source.RelativeMultiTones.RelativeFrequencyIndexes #will be a System.Int32[]
        freq_multipliers = np.full(net_idxs.Length, 0, dtype="int64")
        #there isn't an easy way to typecast so we will do this element-wise
        for i in range(0, net_idxs.Length):
            freq_multipliers[i] = net_idxs[i]

        #build the frequency array (these are the frequencies actually being excited)
        tone_freqs = freq_multipliers*offset_frequency + center_frequency

        #create the excitation frequencies (assumes grid is uniform and tone frequencies go to the edges of the excitation grid)
        excitation_freqs = np.arange(np.min(tone_freqs), np.max(tone_freqs), offset_frequency)
        excitation_freqs = np.append(excitation_freqs, np.max(tone_freqs))
        # excitation_freqs.append(np.max(tone_freqs))

        #now generate the grid modifier (throw error if any frequencies are off the excitation grid)
        # source_grid_mod = atg.ExistsAtFrequencies(excitation_freqs, interpret_as_absolute=True, error_on_mismatch=True)
        source_grid_mod = atg.ExistsAtFrequencies(src_freqs, interpret_as_absolute=True, error_on_mismatch=True)
        excitation_grid_mod = atg.ExistsAtFrequencies(tone_freqs, interpret_as_absolute=True, error_on_mismatch=True)

        #There isn't a need to create a source as the measurement grid will be used as the source
        grid_source = None

        #Build the generator
        source_grid_gen = atg.GridGenerator(grid_source, source_grid_mod)
        excitation_grid_gen = atg.GridGenerator(grid_source, excitation_grid_mod)

        #now generate the new excitation grid
        new_source_grid = atg.Grid.generate("excitation_"+name, using=source_grid_gen, on=self.measurement_grid)

        #now generate the new tone grid
        new_excitation_grid = atg.Grid.generate("tone_"+name, using=excitation_grid_gen, on=new_source_grid)

        #return the generated grid 
        return new_source_grid, new_excitation_grid

    """
    Methods for Network Emulation Setup
    Once tuners have been initialized and setup a further level of abstraction may also be performed by
    combining the following properties: a S-Parameter Network Response, a Tuner Interface, and a Signal. 
    These three properties are combined together to create an Emulated Network object that also requires 
    construction. 

    Emulated networks perform three tasks:
        1) Determine the equivalent S0, T0, and A0 for an equivalent source that emulates the behavior of a 
           physical network (from the dut's perspective).
        2) De-embed the a/b waves from the internal measurement plane, i.e. at the DUT, to the generator/load
           load plane.
        3) Manage the mapping between the different grids (Measurement, Tuner, and Signal)
    """

    def create_emulated_source_network(self, with_signal:Union[None, mts.MultitoneSignal]=None)->EmulatedNetwork:
        #Create an emulated source network (assumes port 2 is on dut)

        #get the tuner
        source_tuner = self.tuners[0]

        if with_signal is None:
            #pull in the stored source signal
            sig = self.__source_signal
        else:
            #make sure that it is a multi-tone signal
            assert isinstance(with_signal, mts.MultitoneSignal), "Provided signal is not a multitone signal definition."
            #TODO: Make sure the signal is compatible with the grid
            #set the signal
            sig = with_signal
        
        #create the source network
        self.__source_net = EmulatedNetwork.thru(source_tuner, sig, self.measurement_grid)

        #return the source network
        return self.__source_net

    def create_emulated_load_network(self, with_signal:Union[None, mts.MultitoneSignal]=None)->EmulatedNetwork:
        #Create an emulated load network (assumes port 2 is on dut)

        #get the tuner
        load_tuner = self.tuners[1]
        
        #set the signal
        if with_signal is None:
            #the load signal will be assumed to be a null signal
            sig = mts.MultitoneSignal.null_signal(load_tuner.grid)
        else:
            assert isinstance(with_signal, mts.MultitoneSignal), "Provided signal is not a multitone signal definition."
            #TODO: Make sure the signal is compatible with the grid
            #set the signal
            sig = with_signal

        #create the load network
        self.__load_net = EmulatedNetwork.thru(load_tuner, sig, self.measurement_grid)

        #return the load network
        return self.__load_net

    ## New Core Methods
    def move_to(self, 
                load_gamma:Union[np.ndarray, None]=None,        #load reflection coefficient
                load_net_data:Union[np.ndarray, None]=None,     #load network data fx2x2
                source_gamma:Union[np.ndarray, None]=None,      #source reflection coefficient 
                source_net_data:Union[np.ndarray, None]=None,   #source network data fx2x2
                net_data_type:str="python",                     #may be python or matlab    
                signal_power:Union[float,None]=None,            #signal power (in W)
                iterations:int=20,                              #number of outter loop iterations    
                err_thresh:float=-20,                           #error threshold for gammas in outter loop
                sub_iterations:int=20,                          #number of inner loop iterations
                sub_err_thresh:float=-20,                       #error threshold for gammas in inner loop 
                debug:bool=False                                #set to true to start debugging plots
                )->tuple[bool, int, float, np.ndarray, float, np.ndarray, float]:                         
        #Have a tuner move to a new impedance

        #make sure that the tuners are available
        if not self.tuners_ready:
            raise RuntimeError("Tuners are not ready yet. Please make sure that the setup procedure")
        
        #get the tuners from the list
        source_tuner = self.tuners[0]
        load_tuner   = self.tuners[1]

        #determine the availability of the load and source networks
        source_net_available = not (self.__source_net is None)
        load_net_available   = not (self.__load_net is None)

        """
        Set the tuner gammas. 
        """

        #get the source gamma 
        if source_net_available:
            #Make sure that the tuner is associated with the source network
            assert source_tuner is self.__source_net.tuner, "Source network and associated tuner are not the same."

            #If a source network is available we will use that instead
            if not (source_net_data is None):
                # self.__source_net.import_spdata(source_net_data, format_type=net_data_type)
                self.__source_net.set_network(source_net_data, data_source=net_data_type)
            else:
                #make sure that the tuner response has been updated from the network
                self.__source_net.update_tuner()
        else:
            #use direct source gamma updates on the tuner
            if source_gamma is None:
                #set the target gamma of the source tuner
                source_tuner.target_gamma = self.__set_source_gamma
            else:
                #use the gamma provided 
                source_tuner.target_gamma = source_gamma

        #get the load gamma
        if load_net_available:
            #Make sure that the tuner is associated with the load network
            assert load_tuner is self.__load_net.tuner, "Load network and assiciated tuner are not the same."

            #If a source network is available we will use that instead
            if not (load_net_data is None):
                # self.__source_net.import_spdata(source_net_data, format_type=net_data_type)
                self.__load_net.set_network(load_net_data, data_source=net_data_type)
            else:
                #make sure that the tuner response has been updated from the network
                self.__load_net.update_tuner()
        else:
            #use direct source gamma updates on the tuner
            if load_gamma is None:
                #set the target gamma of the source tuner
                load_tuner.target_gamma = self.__set_load_gamma
            else:
                #use the gamma provided 
                load_tuner.target_gamma = load_gamma

        #update the signal to play from source 1 
        if not signal_power is None:
            #update the power level 
            #(this will update the signal already in the tuner)
            self.__source_signal.power = signal_power

        #configure the return values
        success = False
        gamma_s_meas = np.full_like(self.__set_source_gamma, 0, dtype=np.dtype("complex"))
        gamma_s_err = 0.0 
        gamma_l_meas = np.full_like(self.__set_load_gamma, 0, dtype=np.dtype("complex"))
        gamma_l_err = 0.0
        gamma_err = 0.0
        ios = iterations

        #NEW: Configure the plotter to show the new data
        if isinstance(source_tuner.plotter, LinkedGammaDebugPlotter): 
            #configure the plotter if it is linked plotter for tracking the count
            source_tuner.plotter.set_tuner_iterations(0, sub_iterations)
            source_tuner.plotter.set_tuner_iterations(1, sub_iterations)
            #set the maximum number of outter tuner iterations
            source_tuner.plotter.total_iterations = iterations

        #DEBUG: Set the target gammas for both the source and load tuners (fixes target not showing up in load)
        # if not (source_gamma is None):
        #     source_tuner.target_gamma = source_gamma
        
        # if not (load_gamma is None):
        #     load_tuner.target_gamma = load_gamma
        
        
        
        #now iteratively update the load and source tuners
        for idx in range(0, iterations):
            #update the source gamma 
            source_tuner.move_to(iterations=sub_iterations, error_threshold=sub_err_thresh, debug=debug)

            #update the load gamma
            load_tuner.move_to(iterations=sub_iterations, error_threshold=sub_err_thresh, debug=debug)
            
            #make sure to update the values of the source tuner
            source_tuner.update()
            
            #get the gamma errors from each tuner
            gamma_s_err = source_tuner.gamma_error
            gamma_l_err = load_tuner.gamma_error

            #compute the total error
            gamma_err = 10 * np.log10(gamma_s_err + gamma_l_err)

            #handle the break condition 
            if gamma_err <= err_thresh:
                success = True
                ios = idx + 1
                break
        
        #convert source and load errors to db form 
        gamma_s_err = 10 * np.log10(gamma_s_err)
        gamma_l_err = 10 * np.log10(gamma_l_err)

        #NEW: Reset the plotter
        if isinstance(source_tuner.plotter, LinkedGammaDebugPlotter): 
            source_tuner.plotter.reset_counts()

        #return the final values
        return success, ios, gamma_err, gamma_s_meas, gamma_s_err, gamma_l_meas, gamma_l_err 

    """
    New Property Methods
    """
    @property
    def tuners_ready(self)->bool:
        return (self.__tuners_available     and  #tuner interfaces need to be created
                self.__tuners_setup         and  #tuners must be setup
                self.source1.OutputEnabled  and  #source one must be on
                self.source2.OutputEnabled)      #source two must be on

    @property
    def source_network(self)->Union[EmulatedNetwork,None]:
        #The stored source network shouldn't be directly accessed
        return self.__source_net    
    
    @property
    def load_network(self)->Union[EmulatedNetwork,None]:
        #The stored load network shouldn't be directly accessbable 
        return self.__load_net
    
    @property 
    def source_grids(self):
        return self.__source_grids
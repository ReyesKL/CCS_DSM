from __future__ import print_function
import builtins
import os
import sys
import time
import clr

import datetime
import operator
import logging

# if this line does not work than you have problem with pythonnet installation - see the instructions
import System
from System import String
from System.Collections import *
verbose = False
# from ctypes import *
"""
This is a file designed to handle the dll imports because it is a lot of code overhead
May not be "best practice" and need improvements but its a start 
"""


def print(*args, **kwargs):
    try:
        vb = kwargs.pop("verbose")
    except KeyError:
        vb = True
    if vb:
        builtins.print(*args)



""" define base directories """
# +++ ATTENTION: Adapt library directory before use!
# It has to point to directory at PC, at which the script is executed! +++
ccs_libs_dir = r'C:\CCS Releases\CCS 1.5.0d20beta1'
# ccs_libs_dir = r"C:\Users\Reyes\OneDrive - UCB-O365\Documents\Research\NI Goodies\VSTPlus_sw"
print("\n# CCS Library Root Directory: {}".format(ccs_libs_dir), verbose=verbose)

# +++ ATTENTION: Adapt data storage directory before use! It has to point to directory on PXINACE! +++
# inputWaveformsDir = r"C:\Users\Reyes\PycharmProjects\VST_System"
input_waveforms_dir = r"C:\Waveforms"
# you have to mirror the content of the waveform dir or explicitly provide the list of the file names
input_waveforms_dir_local = r"/"
print("\n# Input Waveforms Directory: {}".format(input_waveforms_dir), verbose=verbose)

# +++ ATTENTION: Adapt data storage directory before use! It has to point to directory on controlling PC! +++
data_storage_dir = r"/"
print("\n# Data Storage Directory: {}".format(data_storage_dir), verbose=verbose)

''' configure logger'''
# logging.basicConfig(filename=os.path.join(data_storage_dir, 'example_single.log'), level=logging.DEBUG)
# logger = logging.getLogger(__name__)

''' Load CCS assemblies / libraries '''

print("\n# walking CCS library directory recursively generation a file list of CCS assemblies / libraries bottom "
      "to top...", verbose=verbose)

bin_file_paths = list()
bin_names = list()
bin_roots = list()
for root, dirs, files in os.walk(ccs_libs_dir, False):
    # ignore
    root_added_to_list = False
    for file in files:
        if file.endswith(".dll"):
            tmp_path = os.path.join(root, file)
            bin_file_paths.append(tmp_path)
            bin_names.append(file.replace(".dll", ""))
            if not root_added_to_list:
                bin_roots.append(root)
                root_added_to_list = True

print("\n# Temporarily appending PYTHONPATH with binary root paths to be used with pythonnet...", verbose=verbose)
for root in bin_roots:
    if root not in sys.path:
        sys.path.append(str(root))

print("\n# Loading .NET binaries...", verbose=verbose)
for binName in bin_names:
    try:

        clr.AddReference(binName)
        print(" -> PASS - Loaded assembly: {}".format(binName), verbose=verbose)
    except Exception as ex:
        print(
            "-> FAIL - Could not load assembly {}! It very likely contains unmanaged code which is not supported "
            "by pythonnet. Try loading via ctypes instead.".format(
                binName), verbose=verbose)
        # try:
        #     cdll.LoadLibrary(binName+'.dll')
        # except:
        #     print("coult not load via ctypes")
        print("    Internal Error: {}".format(ex), verbose=verbose)

print("\n# Initializing Third Party License Provider...", verbose=verbose)
from NationalInstruments.HFPlatform.Utils import LicenseProvider

LicenseProvider.InitializeThirdPartyLicensing()

print("\n# Initializing .NET Remoting for CCS...", verbose=verbose)
from NationalInstruments.HFPlatform.RemotingLibrary import ClientServices
from NationalInstruments.HFPlatform.RemotingLibrary import IRemoteServer
from NationalInstruments.HFPlatform.Instruments import RFAnalyzer, RFSource, AmplifiedRFSource
from NationalInstruments.HFPlatform.VSTLSNA import RFSource_VST
from NationalInstruments.HFPlatform.Instruments.DC import \
    DCAnalyzer, DCSource, DCSourceMode, DCPolarity, DCLimits, DCGabaritLimits
from NationalInstruments.HFPlatform.Utils import Data, Range
from NationalInstruments.HFPlatform.Signal import Quantity
from NationalInstruments.HFPlatform.Math import AlignerForPeriodicModulatedSignals
from Extreme.Mathematics import DoubleComplex

print("Successfully loaded DLLs for remote instrument control \n")

"""
Common Utilities for the VST
"""

#Imports here 
import os
import numpy as np
from typing import Union
from pathlib import Path
import sys, os

sys.coinit_flags = 2

import tkinter as tk
from tkinter import filedialog, messagebox
import scipy.signal as signal

#VST libraries that this module depends on 
from lib.active_tuner_grid_lib import Grid
from lib.multitone_signal_lib import MultitoneSignal

#For demo purposes only
#get the current directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
#get the parent directory
parent_dir = os.path.dirname(current_dir)
#add it to the python path
sys.path.append(parent_dir)

def get_demo_test_signal(sig_idx:int=0, tone_spacing:float=1e6):
    #import the vst signal class
    from vst_signal import signal
    #set the excitation signal
    FILE_PATH = Path(current_dir, r"../CCS_Measure/demos", r"assets")
    excitation_file_path = FILE_PATH
    excitation_file_name = Path(r'300_tone_random_phase_pars_wider_grid_wider_notch_decimated_10.h5')
    excitation_file = Path.joinpath(excitation_file_path, excitation_file_name)
    signals = signal.from_h5(excitation_file)

    #set the signal tone spacing
    for sig in signals:
        sig.set_tone_spacing(tone_spacing)

    #set the test signal
    return signals[sig_idx]

def apply_pyplot_bugfix():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#For working with complex numbers
def polar2complex(magIn:Union[float,np.ndarray[float]], phaseIn:Union[float,np.ndarray[float]], phase_is_degrees:bool=True)->Union[complex,np.ndarray[complex]]:
    #convert from degrees to radians
    if phase_is_degrees:
        phaseIn = np.deg2rad(phaseIn)
    #return the new complex number
    return magIn * (np.e**(1j * phaseIn))

#File functions
def fileparts(fname:str)->tuple[str, str, str]:
    #Takes in a file and returns the directory, name, and extension
    #of the file

    #get the directory and file name
    directory, filename = os.path.split(fname)
    _, extension = os.path.splitext(filename)

    #return all parts of the file
    return directory, filename, extension

def fullfile(*argv:str)->str:
    #Builds a full file from parts
    
    #use os.path.join
    return os.path.join(*argv)

def isfolder(folderName:str, can_write:bool=False, can_read:bool=False)->bool:
    #Check if a folder exists with optional write and read checks

    #initialize the result
    result = True

    #check if the folder exists
    if isinstance(folderName, str):
        #check if the directory exists
        result = os.path.isdir(folderName) and result
    else:
        raise TypeError("Folder name must be a string.")
    
    #check if we can write to it
    if can_write:
        result = os.access(folderName, os.W_OK)
    
    #check if we can read from it 
    if can_read:
        result = os.access(folderName, os.R_OK)

def isfile(fileName:str, can_write:bool=False, can_read:bool=False)->bool:
    #check if a file exists and with optional read and write checks

    #initialize the result
    result = True

    #check if the file exists
    if isinstance(fileName, str):
        result = os.path.exists(fileName) and result
    else:
        raise TypeError("File name must be a string.")
    
    #check if we can write to it
    if can_write:
        result = os.access(fileName, os.W_OK)
    
    #check if we can read to it
    if can_read:
        result = os.access(fileName, os.R_OK)

def uigetfile(title:Union[str,None]=None,
              initialdir:Union[str,None]=None,
              initialfile:Union[str,None]=None,
              filetypes=[("All Files","*")],
              defaultextension=None,
              multiple:bool=False):
    #Opens UI to allow users to select file path

    #create a new tk window
    root = tk.Tk()
    #bring tthe root to the top
    root.attributes("-topmost", True)
    #hide the root window
    root.withdraw()
    #open the file dialog
    file_path = filedialog.askopenfilename(
        parent=root,
        title=title,
        initialdir=initialdir,
        initialfile=initialfile,
        filetypes=filetypes,
        defaultextension=defaultextension,
        multiple=multiple
    )
    #now destroy the root window
    root.destroy()
    #return the file path
    return file_path

def uiputfile(title:Union[str,None]=None,
              initialdir:Union[str,None]=None,
              initialfile:Union[str,None]=None,
              filetypes=[("All Files","*")],
              defaultextension=None):
    #Opens UI for file saving 

    #create a new tk window
    root = tk.Tk()
    #bring the root to the top
    root.attributes("-topmost",True)
    #hide the root window
    root.withdraw()
    #open the file dialog
    file_path = filedialog.asksaveasfilename(
        parent=root,
        title=title,
        initialdir=initialdir,
        initialfile=initialfile,
        filetypes=filetypes,
        defaultextension=defaultextension)
    #close the window
    root.destroy()
    #return the file path
    return file_path

def uigetdir(title:Union[str,None]=None,
            initialdir:Union[str,None]=None,
            mustexist:bool=True):
    """
    Get the directory from the user interface
    """

    #create a new tk window
    root = tk.Tk()
    #bring the root to the top
    root.attributes("-topmost",True)
    #hide the root window
    root.withdraw()
    #open the file dialog
    directory_path = filedialog.askdirectory(
        parent=root,
        title=title,
        initialdir=initialdir,
        mustexist=mustexist)
    #close the window
    root.destroy()
    #return the path to the directory
    return directory_path

def uiyesno(title:Union[str,None]=None, dialog:Union[str,None]=None):

    #create a new window
    root = tk.Tk()
    #bring the root to the top
    root.attributes("-topmost",True)
    #hide the root window
    root.withdraw()
    #create messagebox
    response = messagebox.askyesno(title, dialog)
    #close the window
    root.destroy()
    #return the response
    return response

#for power conversion
#Function for power conversion
def dbm2w(dbmIn:float)->float: 
    return 10**((dbmIn-30)/10)

def w2dbm(wIn:float)->float: 
    return 10 * np.log10(1e3 * np.abs(wIn))

def db(x:Union[float, np.ndarray[float]])->Union[float, np.ndarray[float]]:
    return 10*np.log10(np.abs(x))

def db2mag(x:Union[float, np.ndarray[float]])->Union[float, np.ndarray[float]]:
    return 10**(x / 10)

def complex2rad(x:Union[complex, np.ndarray[complex]])->Union[float, np.ndarray[float]]:
    return np.angle(x)

def complex2deg(x:Union[complex, np.ndarray[complex]])->Union[float, np.ndarray[float]]:
    return np.rad2deg(complex2rad(x))

#helpers for VST specific elements
def get_vst_source_multitone_data(source)->tuple[float, np.ndarray[int], np.ndarray[float], np.ndarray[float]]:
    #Pulls in the relative multitone data from the provided VST source

    #get the data from the source
    rel_amps = np.array(list(map(float, source.RelativeMultiTones.RelativeAmplitudes)))
    rel_phases = np.array(list(map(float, source.RelativeMultiTones.RelativePhases)))
    tone_idxs = np.array(list(map(int, source.RelativeMultiTones.RelativeFrequencyIndexes)))
    output_level = source.OutputLevel

    return output_level, tone_idxs, rel_amps, rel_phases

#Useful tools for manipulating power waves 
def change_ref_impedance(ai, bi, zi, zk)->tuple[np.ndarray[complex], np.ndarray[complex]]:
    """
    Change the reference impedance of the power waves, ai and bi, from zi to zk.
    """
    
    #initialize the return variables
    ak = np.zeros_like(ai, dtype="complex"); bk = np.zeros_like(bi, dtype="complex")

    #get the number of points in ai and bi and make sure that they are the same
    if not ai.size == bi.size:
        raise IndexError(f"Size of ai {ai.size} does not match the size of bi {bi.size}")
    else:
        N = ai.size

    #also check Zi
    if isinstance(zi, (int,float,complex)):
        zi = np.full((N,), zi, dtype="complex")
    elif isinstance(zi, np.ndarray):
        #check the size
        if not zi.size == N:
            raise IndexError(f"Size of zi {zi.size} must match that of the power waves {N}")
        #make sure that the type is complex
        zi = np.array(zi, dtype="complex")
    else:
        raise TypeError("Unrecognized type for zi")

    #also check Zk
    if isinstance(zk, (int,float,complex)):
        zk = np.full((N,), zk, dtype="complex")
    elif isinstance(zk, np.ndarray):
        #check the size
        if not zk.size == N:
            raise IndexError(f"Size of zk {zk.size} must match that of the power waves {N}")
        #make sure that the type is complex
        zk = np.array(zk, dtype="complex")
    else:
        raise TypeError("Unrecognized type for zk")


    #build the system to solve
    A = np.zeros((N,2,2), dtype="complex")
    A[:,0,0] = np.conj(zk[:]); A[:,0,1] = zk[:]
    A[:,1,0] = 1; A[:,1,1] = -1

    #build Ki and Kk
    pi = np.sign(np.real(zi)); Ki = pi / np.sqrt(np.real(zi))
    pk = np.sign(np.real(zk)); Kk = pk / np.sqrt(np.real(zk))

    #build the b vector
    b = np.zeros((N,2,1), dtype="complex")
    b[:,0,0] = (Ki/Kk)*(np.conj(zi)*ai + zi*bi)
    b[:,1,0] = (Ki/Kk)*(ai - bi)

    #now solve the system of equations
    x = np.matmul(np.linalg.inv(A), b)

    #now broadcast to the return vectors
    ak[:] = x[:,0,0]; bk[:] = x[:,1,0]

    #return
    return ak, bk

#function for converting from voltage to power waves 
# (bench doesn't measure power waves but voltage waves)
def v_to_p_waves(vp,vm,z0):
    #simple function for converting to power waves
    a = vp / np.sqrt(np.abs(np.real(z0)))
    b = vm / np.sqrt(np.abs(np.real(z0)))
    #now return the power wave definitions
    return a, b

#function for converting from gamma to z
def gamma_to_z(gamma, z0=50.0):
    return z0*(1 + gamma)/(1 - gamma)


"""
Linearity functions
"""

def get_td_waveform(data, freqs):
    """
    Get the time domain waveform from the frequency domain waveform.
    
    Parameters:
    data : np.ndarray
        The frequency domain waveform.
    freqs : np.ndarray
        The frequencies corresponding to the waveform.

    Returns:
    np.ndarray
        The time domain waveform.
    """
    #check that the data is a 1D array
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array.")

    #make sure that the number of frequencies is odd
    if np.mod(len(freqs), 2) == 0:
        raise ValueError("Number of frequencies must be odd.")

    #shift the frequencies to be centered around zero
    freqs = freqs - np.mean(freqs)

    #get the frequency step size
    df = np.abs(freqs[1] - freqs[0])

    #check that the frequencies are evenly spaced
    if not np.all(np.isclose(np.diff(freqs), df)):
        raise ValueError("Frequencies must be evenly spaced.")
    
    #get the number of points in the frequency domain waveform
    N = data.size

    #get the time step size
    dt = 1/(N*df)

    #get the baseband values of the output waveform
    # wave_out = np.fft.ifft(data,N) * N
    wave_out = np.fft.ifft(data, norm="forward")

    #get the time vector
    time = np.arange(N) * dt

    #return the output waveform 
    return time, wave_out

def calc_td_powers(a1, b1, a2, b2, freqs, z0=50.0):
    """
    Calculate the AM-AM conversion from the input and output power waves.
    
    Parameters:
    a1 : np.ndarray
        The input power wave.
    b2 : np.ndarray
        The output power wave.
    freqs : np.ndarray
        The frequencies corresponding to the waveforms.
    z0 : float, optional
        The reference impedance (default is 50.0).

    Returns:
    np.ndarray
        The AM-AM conversion.
    """

    #get the time-domain waveforms
    t, a1t = get_td_waveform(a1/np.sqrt(z0), freqs)
    _, b1t = get_td_waveform(b1/np.sqrt(z0), freqs)
    _, a2t = get_td_waveform(a2/np.sqrt(z0), freqs)
    _, b2t = get_td_waveform(b2/np.sqrt(z0), freqs)


    #get the currents and voltages 
    v1 = (np.conj(z0) * a1t + z0 * b1t) / np.sqrt(np.abs(np.real(z0)))
    i1 = (a1t - b1t) / np.sqrt(np.abs(np.real(z0)))

    v2 = (np.conj(z0) * a2t + z0 * b2t) / np.sqrt(np.abs(np.real(z0)))
    i2 = (a2t - b2t) / np.sqrt(np.abs(np.real(z0)))

    #get pin and pout (assuming the system is matched)
    pin = (v1 * np.conj(i1)) / 2
    pout = (v2 * np.conj(i2)) / 2

    corr = signal.correlate(pin, pout)
    lags = signal.correlation_lags(len(pin), len(pout))
    lag = lags[np.argmax(np.abs(corr))]

    pout = np.roll(pout, -lag)
    
    #return the input power and gain 
    return pin, pout, t

class acpr_manager:

    def __init__(self, reference_signal, measurement_grid, 
                 guard_bandwidth:float = 10e6,
                 adjacent_channel_bandwidth:Union[float, None]=None):
        
        #Set the hidden properties for this object
        self.__ref_sig = None
        self.__ref_channel_mask   = None
        self.__lower_channel_mask = None
        self.__upper_channel_mask = None

        #Set the measurement grid
        if isinstance(measurement_grid, Grid):
            self.__root_grid = measurement_grid
        else:
            raise TypeError("Measurement grid must be of type Grid")
       
        #set the guard bandwidths
        self.guard_bandwidth = guard_bandwidth
        self.adjacent_channel_bandwidth = adjacent_channel_bandwidth

        #now set the reference signal
        self.reference_signal = reference_signal

    def __build_channel_masks(self):

        #get the signal frequencies 
        sig_freqs = self.reference_signal.grid.freqs
        f_sig_low = sig_freqs[0]; f_sig_high = sig_freqs[-1]

        #now identify the inner frequencies from the gaurd band requirements 
        f1b = f_sig_low - self.guard_bandwidth; f2a = f_sig_high + self.guard_bandwidth

        #identify the bandwidth of the adjacent channels
        if self.adjacent_channel_bandwidth is None: 
            channel_bandwidth = f_sig_high - f_sig_low
        else:
            channel_bandwidth = self.adjacent_channel_bandwidth

        #now identify the outter frequencies from the channel bandwidth requirements
        f1a = f1b - channel_bandwidth; f2b = f2a + channel_bandwidth

        #finally, find the points on the measurement grid between the desired frequencies listed here
        meas_freqs = self.__root_grid.freqs 

        #set the channel points 
        # ref_grid = self.__ref_sig.grid
        # self.__ref_channel_mask   = ref_grid.cast(ref_grid.full_like(True, dtype="bool"), self.__root_grid, off_grid_vals=False, dtype="bool")
        self.__ref_channel_mask   = np.logical_and(f_sig_low <= meas_freqs, meas_freqs <= f_sig_high)
        self.__lower_channel_mask = np.logical_and(f1a <= meas_freqs, meas_freqs < f1b)
        self.__upper_channel_mask = np.logical_and(f2a <  meas_freqs, meas_freqs < f2b)

    def calculate(self, meas_power:np.ndarray[float]):
        #Calculate the ACPR from measured spectral power and returns in dB

        #get the data from each of the three channels
        p_lower = np.sum(meas_power[self.__lower_channel_mask]);    lower_size = self.__lower_channel_mask.size
        p_upper = np.sum(meas_power[self.__upper_channel_mask]);    upper_size = self.__upper_channel_mask.size
        p_ref   = np.sum(meas_power[self.__ref_channel_mask]);      ref_size = self.__ref_channel_mask.size

        #normalize the reference power right out of the gate to avoid redundant calcs
        p_ref_norm = (p_ref / ref_size)

        #calculate the normalized lower acpr 
        acpr_lower = (p_lower / lower_size) / p_ref_norm
        acpr_upper = (p_upper / upper_size) / p_ref_norm
        acpr_full  = ((p_lower + p_upper) / (lower_size + upper_size)) / p_ref_norm

        #return the results of the calculation 
        return db(acpr_lower), db(acpr_upper), db(acpr_full)

    @property 
    def reference_signal(self)->MultitoneSignal:
        return self.__ref_sig
    
    @reference_signal.setter
    def reference_signal(self, new_val:MultitoneSignal):
        #setter for the reference signal
        if not isinstance(new_val, MultitoneSignal):
            raise TypeError("New reference signal must be a multitone signal object")
        #set the signal
        self.__ref_sig = new_val
        #rebuild all channel masks
        self.__build_channel_masks()

class imd3_manager:
    def __init__(self, reference_signal:MultitoneSignal, measurement_grid:Grid):
        
        #create the private properties for this class
        self.__ref_sig      = None
        self.__root_grid    = None
        self.__tone_indices = None

        #Set the measurement grid
        if isinstance(measurement_grid, Grid):
            self.__root_grid = measurement_grid
        else:
            raise TypeError("Measurement grid must be of type Grid")

        #set the reference signal
        self.reference_signal = reference_signal 
    
    def update_tone_locations(self):
        Grid.index()
        #get the signal indices on the root grid
        rt = self.__root_grid

        #get the signal tones on the root grid
        sig_idx = rt.cast_index(self.__ref_sig.grid, about_center=True)

        #determine the imd indices 
        imd_idx = np.ndarray([sig_idx[0]*2 - sig_idx[1], sig_idx[1]*2 - sig_idx[0]], dtype="int")

        #get the tone indices 
        self.__tone_indices = np.sort(np.concatenate((sig_idx, imd_idx)))
    
    def calculate(self, meas_power:np.ndarray[float])->tuple[float, float]:
        #get the powers at the indices provided
        meas_power = meas_power[self.__tone_indices]

        #calculate the lower and upper imd
        imd_lower = meas_power[0] / meas_power[1]
        imd_upper =meas_power[3] / meas_power[2]

        #return the measured values
        return db(imd_lower), db(imd_upper), None

    @property
    def reference_signal(self)->MultitoneSignal:
        return self.__ref_sig

    @reference_signal.setter
    def reference_signal(self, new_val):
        if not isinstance(new_val, MultitoneSignal):
            raise TypeError("Provided object is not a MultitoneSignal object.")
        elif not new_val.num_tones == 2:
            raise ValueError("Provided signal must be a 2-tone signal")
        
        #set the signal
        self.__ref_sig = new_val

        #run the calculate imd's operation
        self.update_tone_locations()

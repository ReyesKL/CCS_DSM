
#imports
from pathlib import Path
import numpy as np
from typing import Union, Any
import warnings
import skrf as rf

#for file dialog method
import tkinter as tk
from tkinter import filedialog

from numpy import floating

from lib.vst_signal import signal
import sys, os
#get the current directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
#get the parent directory
parent_dir = os.path.dirname(current_dir)
#add it to the python path
sys.path.append(parent_dir)




#Function for importing a demo test signal for debugging
#and demonstration
def get_demo_test_signal(sig_idx:int=0, tone_spacing:float=1e6):
    #import the vst signal class

    #set the excitation signal
    FILE_PATH = Path(current_dir, r"../CCS_Measure/demos/assets")
    excitation_file_path = FILE_PATH
    excitation_file_name = Path(r'300_tone_random_phase_pars_wider_grid_wider_notch_decimated_10.h5')
    # excitation_file_name = Path(r'2_tone_varying_bw.h5')
    excitation_file = Path.joinpath(excitation_file_path, excitation_file_name)
    signals = signal.from_h5(excitation_file)

    #set the signal tone spacing
    for sig in signals:
        sig.set_tone_spacing(tone_spacing)

    #set the test signal
    return signals[sig_idx]

#Functions for generating gamma targets to demonstrate
def single_point_gamma(location:complex, for_tuner, check_gamma:bool = True):
    #the number of points for the target gamma should be the
    #same size as the the tuner grid

    #automatic typecasting from float and int to complex
    if isinstance(location,(int, float)):
        location= complex(location)

    #make sure it's the right type
    if not isinstance(location,(complex)):
        raise TypeError("Location must be a complex number.")
    elif check_gamma and (np.abs(location) > 1):
        warnings.warn("Target reflection coefficient magnitude > 1")

    #now make the target gamma
    return np.full(for_tuner.grid.size, location)

def spiral_gamma_target(for_tuner, rotation:float=0.0, offset:complex=0, 
                         num_turns:float=3.0, radius:float=1.0, rotation_in_degrees:bool=True, 
                         check_gamma:bool=True):
    #create a spiral gamma target 
    #get the offset gamma values
    offset_gammas = single_point_gamma(offset, for_tuner, check_gamma=False)
    
    #generate the angles
    phi_start   = 0
    phi_stop    = float(num_turns) * 2 * np.pi
    phis        = np.linspace(phi_start, phi_stop, for_tuner.grid.size)

    #generate the magnitudes
    mags = radius * (phis / phi_stop)

    #now generate the final gammas
    gamma_vals = polar2complex(mags, phis, phase_is_degrees=False)
    rot = polar2complex(1, rotation, phase_is_degrees=rotation_in_degrees)

    #apply rotation and offset
    gamma_vals = (gamma_vals * rot) + offset_gammas

    #return the final gamma values
    return gamma_vals

def arc_gamma_target(for_tuner, rotation:float=0.0, offset:complex=0, 
                         radius:float=1.0, in_degrees:bool=True, 
                         angle_start:float=-180, angle_stop:float=180,
                         check_gamma:bool=True):
    
    #create a spiral gamma target 
    #get the offset gamma values
    offset_gammas = single_point_gamma(offset, for_tuner, check_gamma=False)
    
    #start angle
    phi_start = angle_start
    if in_degrees:
        phi_start   = np.deg2rad(phi_start)

    #stop angle
    phi_stop    = angle_stop
    if in_degrees:
        phi_stop = np.deg2rad(phi_stop)

    phis = np.linspace(phi_start, phi_stop, for_tuner.grid.size)

    #generate the magnitudes
    mags = radius

    #now generate the final gammas
    gamma_vals = polar2complex(mags, phis, phase_is_degrees=False)
    rot = polar2complex(1, rotation, phase_is_degrees=in_degrees)
    
    #apply rotation and offset
    gamma_vals = (gamma_vals * rot) + offset_gammas

    #issue warning if anything leaves the smith chart
    if check_gamma and np.any(np.abs(gamma_vals) > 1):
        warnings.warn("One or more values of helical gamma left the smith chart")

    #return with the offset and final rotation
    return gamma_vals

def cardioid_gamma_target(for_tuner, rotation:float=0.0, offset:complex=0.0, 
                          radius:float=1.0, rotation_in_degrees:bool=True, 
                          check_gamma=True):
    
    #get the offsets
    offset_gammas = single_point_gamma(offset, for_tuner, check_gamma=False)

    #create a ardioid gamma target
    phi_start = 0
    phi_stop  = 2 * np.pi
    phis      = np.linspace(phi_start, phi_stop, for_tuner.grid.size)

    #create the radius
    mags = (1 - np.cos(phis)) * radius

    #now convert to gamma values
    gamma_vals = polar2complex(mags, phis, phase_is_degrees=False)
    rot = polar2complex(1, rotation, phase_is_degrees=rotation_in_degrees)

    #apply rotation and offset
    gamma_vals = (gamma_vals * rot) + offset_gammas
    
    #check if any values would leave the smith chart
    if check_gamma and np.any(np.abs(gamma_vals) > 1):
        warnings.warn("One or more values of helical gamma left the smith chart")
    
    #return the final values
    return gamma_vals

def random_gamma_target(for_tuner, offset:complex=0.0, radius:float=1.0, check_gamma=True):
    #generate a random group of gamma points

    #get the offsets
    offset_gammas = single_point_gamma(offset, for_tuner, check_gamma=False)

    #generate random phases and magnitudes
    mags = np.random.uniform(0, 1, for_tuner.grid.size) * radius
    phis = np.random.uniform(-180, 180, for_tuner.grid.size)

    #generate the gamma values
    gamma_vals = polar2complex(mags, phis, phase_is_degrees=True) + offset_gammas

    #check if any values would leave the smith chart
    if check_gamma and np.any(np.abs(gamma_vals) > 1):
        warnings.warn("One or more values of helical gamma left the smith chart")

    #return the final gamma values
    return gamma_vals

def tl_network_gamma_target(for_network, electric_length:float, at_freq:float, Zc:float, Z0:float=50.0, in_degrees:bool=True):
    #Generates the s-parameters for a transmission line 
    
    #get the network frequencies
    freqs = for_network.freqs

    #build the electric length vector
    el = electric_length * freqs / at_freq

    #convert to radians
    if in_degrees:
        el = np.deg2rad(el)

    #generate the ABCD parameters
    A = np.cos(el)
    B = 1j * Zc * np.sin(el)
    C = (1j * np.sin(el) / Zc)  
    D = np.cos(el)

    a = np.full((freqs.size,2,2),0,dtype=np.dtype('complex'))

    #set the parameters
    a[:,0,0] = A; a[:,0,1] = B
    a[:,1,0] = C; a[:,1,1] = D

    #convert to s-parameters and return 
    return rf.a2s(a,z0=Z0)



#Function for converting polar form to complex form
def polar2complex(magIn:Union[float,np.ndarray[float]], phaseIn:Union[float,np.ndarray[float]], phase_is_degrees:bool=True)->Union[complex,np.ndarray[complex]]:
    #convert from degrees to radians
    if phase_is_degrees:
        phaseIn = np.deg2rad(phaseIn)
    #return the new complex number
    return magIn * (np.e**(1j * phaseIn))


def complex2polar(zIn:Union[complex,np.ndarray[complex]], phase_is_degrees:bool=True)-> tuple[
    float, Union[floating[float], float]]:
    mag = np.abs(zIn)
    angle = np.angle(zIn)
    if phase_is_degrees:
        angle = np.rad2deg(angle)
    return mag, angle

#Function for power conversion
def dbm2w(dbmIn:float)->float: 
    return 10**((dbmIn-30)/10)

def w2dbm(wIn:float)->float: 
    return 10 * np.log10(1e3 * np.abs(wIn))

#Function for getting an sparameter file
def ui_get_touchstone():
    #create a new tk window
    root = tk.Tk()
    #bring the root to the top
    # root.lift()
    root.attributes("-topmost",True)
    #hid the root window
    root.withdraw()
    #open the file dialog
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select a touchstone file",
        filetypes=[("Touchstone File", "*.s2p *.s3p *.s4p *.s5p *.s6p *.s7p *.s8p *.s9p *.snp"),
                   ("Text File", "*.txt")]
    )
    #now destroy the root window
    root.destroy()
    #Return the file path
    return file_path
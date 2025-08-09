"""
Title: Waveform Generator
Author: Paul Flaten
Date: 8/9/2025
Description: TODO
"""


#Perform imports for signal creation 
import numpy as np
from multitone_signal_lib import MultitoneSignal
from active_tuner_grid_lib import Grid, GridGenerator, OrMask
from typing import Union, Literal



"""
Multitone signal generator
"""


class Multitone_Waveform_Generator:

    def __init__(self, name:str, root_grid:Grid, signal_bandwidth:float, 
                 notch_frac_width:Union[int,float]=0.1, 
                 center_frequency:Union[float,None]=None,
                 signal_period:Union[int,float,None]=None,
                 trials:int=101, 
                 random_amplitude:bool = False,
                 random_phase:bool = True): 
        
        # parameters for this generator
        self.name                = name
        self.signal_bandwidth    = signal_bandwidth
        self.notch_frac_width    = notch_frac_width

        # set the grid properties
        self.root_grid           = root_grid
        self.ref_grid            = None
        self.signal_grid         = None

        # for signal search 
        self.trials              = trials
        self.random_amplitude    = random_amplitude
        self.random_phase        = random_phase

        # for the results of the search 
        self.excitatons          = None
        self.pars                = None

        # for the internal siganl
        self.__signal            = None
        self.__num_tones         = None

        #Initialize the signal period
        if signal_period is None: 
            self.signal_period = root_grid.period
        elif isinstance(signal_period, (float,int)):
            self.signal_period = float(signal_period)
        else:
            raise TypeError("Datatype for signal period must be None, int, or float")

        #Initialize the center frequency
        if center_frequency is None:
            self.center_frequency = root_grid.ref_freq
        elif isinstance(center_frequency,(int,float)) and center_frequency >= 0:
            self.center_frequency = center_frequency


        #build the grids 
        self.build_grids()

        #creat a new set of signals
        self.run()

    def build_grids(self):
        # Step 1: clear all grids
        # self.clear_grids()

        # Step 2: build the reference grid
        self.build_reference_grid()

        # Step 3: build the signal grid 
        self.build_signal_grid()

    def run(self):
        
        #create a new signal from the signal grid 
        self.__num_tones = self.signal_grid.size

        #initialize the excitation structure 
        self.excitatons        = np.zeros((self.trials, self.__num_tones), dtype="complex")
        self.pars              = np.zeros((self.trials,), dtype="float")
        
        #build the new signal 
        self.__signal = MultitoneSignal(self.__num_tones, 
                                        self.signal_grid, 
                                        with_tone_vals = np.ones((self.__num_tones,), dtype="complex"))

        #now get the par from each trial 
        for trial in np.arange(self.trials):
            A0 = self.__get_next_excitation()
            #set the next excitation for the signal 
            self.__signal.a0 = A0
            #now get the corresponding par 
            self.pars[trial] = 10*np.log10(self.__signal.par)
            #save the excitation from this trial
            self.excitatons[trial,:] = A0[:]

    def __get_next_excitation(self): 
        
        #generate the random magnitudes
        if self.random_amplitude: 
            mags = np.random.rand(self.__num_tones)
        else:
            mags = np.ones((self.__num_tones,),dtype="float")

        #generate the random phases
        if self.random_phase:
            phases = np.random.rand(self.__num_tones) * 2 * np.pi
        else:
            phases = np.zeros((self.__num_tones,),dtype="float")

        #generate the random signal amplitudes        
        return mags * np.exp(1j * phases)

    def clear_grids(self):
        pass

    def build_reference_grid(self):
        
        #get the root grid period 
        root_period = self.root_grid.period
        root_res    = self.root_grid.frequency_resolution

        #Handle the case where the reference grid is already set
        if self.ref_grid is not None:
            raise NotImplementedError("Reference grid re-building has not been implemented yet.")
        elif self.signal_period > root_period:
            raise RuntimeError(f"Signal period {self.signal_period} cannot be greatter than root grid period {root_period}")
        
        #Now build the reference grid on the base grid
        signal_period_factor = float(np.round(root_period / self.signal_period))

        #correct the signal period
        self.signal_period = root_period / signal_period_factor

        #also correct the signal bandwdith 
        self.signal_bandwidth = root_res * np.round(self.signal_bandwidth / root_res)

        #get the root grid indices
        root_idx = self.root_grid.index(about_center=True)
        root_freqs = self.root_grid.offset_freqs

        #get the upper and lower relative freqencies of the band
        f_low = -self.signal_bandwidth / 2
        f_high = self.signal_bandwidth / 2

        #now get the minimum and maximum indices on the root grid
        f_low_match = root_idx[np.argmin(np.abs(f_low - root_freqs))]
        f_high_match = root_idx[np.argmin(np.abs(f_high - root_freqs))]

        #the reference grid indices will be the root grid indices multiplied by the signal period factor
        ref_idx = int(signal_period_factor * root_idx)

        #make sure the reference indices fall within the specified bandwidth on the root grid
        ref_idx = ref_idx[np.logical_and(ref_idx >= f_low_match, ref_idx <= f_high_match)]

        #now find the mask we will use to build the reference grid
        ref_mask = np.isin(root_idx, ref_idx)

        #create a new grid generator
        new_generator = GridGenerator(None, OrMask(ref_mask))

        #generate the new reference grid
        self.ref_grid = Grid.generate(f"{self.name}_reference_grid", new_generator, self.root_grid)

        #now shift the grid by the requested center frequency 
        self.ref_grid.shift(float(self.center_frequency - self.ref_grid.ref_freq))

    def build_signal_grid(self):
        
        #build the signal tone grid 
        if not np.mod(self.ref_grid.size,2):
            raise RuntimeError("Something went wrong, the reference grid size is not odd")
        
        #build the reference grid enumeration 
        ref_idx = self.ref_grid.index(about_center=True)

        if self.notch_frac_width:
            #get the notch width from the notch fractional width 
            notch_width = self.notch_frac_width * ref_idx.size

            #make sure that the notch width is even
            if np.mod(notch_width,2):
                notch_width += 1
            
            #get the notch tone indices
            notch_tones = np.arange(-int(notch_width / 2), int(notch_width / 2) + 1)
        else: 
            notch_tones = np.array([])

        #now initialize the signal mask 
        sig_mask = np.full_like(ref_idx, True, dtype="bool")

        #set the notch tones to false
        sig_mask[np.isin(ref_idx, notch_tones)] = False

        #now build the signal grid 
        #create a new grid generator
        new_generator = GridGenerator(None, OrMask(sig_mask))

        #generate the new reference grid
        self.signal_grid = Grid.generate(f"{self.name}_signal_grid", new_generator, self.ref_grid)

    def get_signal_with_par(self, target_par:float, 
                            power:float=0.1, 
                            power_type:Literal["peak", "average"] = "peak", 
                            auto_level:bool=True):
        
        #get the index of the trial with the closest par 
        trial = np.argmin(np.abs(self.pars - target_par))

        #closest found 
        par = self.pars[trial]

        #get the tone values 
        tone_vals = np.zeros((self.__num_tones,), dtype="complex")
        tone_vals[:] = self.excitatons[trial,:]

        #create a new signal to return 
        new_signal = MultitoneSignal(self.__num_tones, self.signal_grid, with_tone_vals=tone_vals,
                                     power=power, power_type=power_type, auto_level=auto_level)
        
        return new_signal, par

    @property
    def min_par(self):
        if self.pars is not None:
            return np.min(self.pars)
        else:
            raise RuntimeError("PARs have not been initialized. Please run the waveform generator.")
    
    @property 
    def max_par(self):
        if self.pars is not None:
            return np.max(self.pars)
        else:
            raise RuntimeError("PARs have not been initialized. Please run the waveform generator.")
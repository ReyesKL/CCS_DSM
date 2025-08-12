import numpy as np
import xarray as xr
from lib.active_tuner_grid_lib import Grid, StaticGridSource, GridGenerator, ExistsAtFrequencies, OrMask
from typing import Union
import os, yaml, h5py
from datetime import datetime

#TODO: At some point in the future, the SignalDefinition class and MultitoneSignal classes should be combined into one entity.

class SignalDefinition:
    @classmethod
    def load(cls, file):
        #load the file from a path
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File does not exist: {file}")
        
        #load the file 
        data_set = xr.open_dataset(file, engine='h5netcdf')

        #get the data 
        frequencies = data_set["frequency"].data
        a0 = data_set["a0"].data

        #pull in the data set attributes
        power            = data_set.power
        power_type       = data_set.power_type
        auto_level       = data_set.auto_level
        z0               = data_set.z0

        #create the new instance
        return cls(frequencies, a0, 
                   power=power, 
                   power_type=power_type, 
                   auto_level=auto_level, z0=z0)

    def __init__(self, frequencies:np.ndarray[float], a0:np.ndarray[complex], 
                 power:float=1e-3, 
                 power_type:str="peak", 
                 auto_level:bool=True, 
                 z0:float=50.0):

        #set the properties of the signal definition 
        self.power        = power
        self.power_type   = power_type
        self.a0           = a0 
        self.auto_level   = auto_level
        self.frequencies  = frequencies
        self.z0           = z0

    def save(self, file):
        #save the signal definition to a file 
        directory = os.path.dirname(file)
        
        #determine if the directory exists
        if not os.path.isdir(directory):
            raise RuntimeError(f"Directory does not exist:{directory}")
        
        #build the tone index
        tone_idx = np.arange(self.a0.size)

        #set the data 
        freq_dat = xr.DataArray(data=self.frequencies, dims=["index"], coords=tone_idx)
        a0_dat = xr.DataArray(data=self.a0, dims=["index"], coords=tone_idx)
        data_dict = dict(frequency=freq_dat, a0=a0_dat)

        #create a dictionary of signal attributes
        attribute_dict = dict("power", self.power, 
                              "power_type", self.power_type,
                              "auto_level", self.auto_level,
                              "z0", self.z0)
        
        #create the xarray dataset 
        data_set = xr.Dataset(data_dict, attrs=attribute_dict)

        #now save the data
        data_set.to_netcdf(file, engine="h5netcdf")

    @property
    def num_tones(self):
        return self.a0.size
        

class MultitoneSignal:
    #methods for building multitone signals
    @classmethod
    def load(cls, file_path:str, to_grid:Grid, name:str = "new_signal"):
        """
        Initial file loader for the test signal 
        """

        #Raise a file not found error if the path specified does not exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find target file: {file_path}")
        elif not os.access(file_path, os.R_OK):
            raise RuntimeError("File found, but cannot be opened for reading.")
        
        #If all checks passed then go ahead and open the file for reading 
        with h5py.File(file_path, "r") as file:
            #get the signal attributes 
            tone_vals = file["A0"][:]
            frequencies = file["Frequencies"][:]
            indices     = file["Tone_Indices"][:]
            num_tones   = tone_vals.size
            power       = file.attrs["Power"]
            power_type  = file.attrs["Power_Type"]
            z0          = file.attrs["Z0"]
            auto_level  = file.attrs["Auto_Level"]
        
        #Check to see if all the frequencies stored in the signal exist on the specified grid
        grid_freqs = to_grid.freqs
        if not np.alltrue(np.isin(frequencies, grid_freqs)):
            raise RuntimeError("Could not map signal to specified grid. Frequencies don't align.")
        
        #perform the signal mapping
        sig_mask = np.isin(grid_freqs, frequencies)

        #build the signal grid
        grid_source = None
        grid_mod = OrMask(sig_mask)
        grid_gen = GridGenerator(grid_source, grid_mod)
        sig_grid = Grid.generate(name, using=grid_gen, on=to_grid)

        #now build the signal
        return cls(num_tones, sig_grid, 
                   power=power, power_type=power_type, with_tone_vals=tone_vals, Z0=z0, auto_level=auto_level)

    @classmethod
    def single_tone(cls, at_frequency:float, on_grid:Grid, grid_name:str="single_tone_signal", power:float=1e-3, phase:float=0, power_type:str="peak", Z0:float=50, auto_level:bool=True, deg:bool=True):
        
        #find the desired frequency on the specified grid
        freq_idx = np.argmin(np.abs(at_frequency - on_grid.freqs))

        #create a mask of the prior grid and set the desired frequency to true
        grid_mask = on_grid.full_like(False,dtype="bool")
        grid_mask[freq_idx] = True

        #now create a simple signal grid
        grid_source = None
        grid_mod    = OrMask(grid_mask)
        grid_gen    = GridGenerator(grid_source, grid_mod)
        sig_grid    = Grid.generate(grid_name,using=grid_gen,on=on_grid)

        #make sure the phase is in the right units
        if deg:
            phase = np.deg2rad(phase)

        #Initialize the center tone
        A0_init = np.exp(1j*phase)
        
        #return the class
        return cls(1, sig_grid, power=power, power_type=power_type, with_tone_vals=A0_init, Z0=Z0, auto_level=auto_level)
    
    @classmethod
    def from_vst_signal(cls, vst_sig, with_grid:Grid, power:float=1e-3, power_type:str="peak", 
                 Z0:float=50, using_ref_grid:Union[Grid,None]=None,
                 auto_level:bool=True, phase_in_deg:bool=True,
                 sig_grid_creation_round_up:bool=True, source_grid_freq_step:Union[float,None]=None,
                 source_grid_center_freq:Union[float,None]=None):
        #allows us to create a new signal from the vst signal class data
        
        """
        First, we need to handle signal sub-grid creation
        """
        if with_grid.size == vst_sig.num_tones:
            #use the provided grid for the signal grid
            sig_grid = with_grid
        else: #we need to construct the signal grid
            #get the slot indices of the vst signal
            slot_idxs = vst_sig.get_grid_slots()
            
            #get the center slot indices
            slt_cntr_idx = np.mean(slot_idxs)

            #handle the round up or round down the center idx
            if sig_grid_creation_round_up:
                slt_cntr_idx = int(np.ceil(slt_cntr_idx))
            else:
                slt_cntr_idx = int(np.floor(slt_cntr_idx))

            #now generate the tone indices
            tone_idxs = slot_idxs - slt_cntr_idx

            #tone frequencies require knowledge of the grid step and center frequency
            if source_grid_freq_step is None:
                #guess the frequency step of provided grid (minimum absolute difference in frequencies)
                source_grid_freq_step = np.min(np.abs(np.roll(with_grid.freqs,1) - with_grid.freqs))

            if source_grid_center_freq is None:
                #guess the center frequency from the provided grid's reference frequency
                source_grid_center_freq = with_grid.ref_freq

            #build the tone frequencies
            tone_freqs = tone_idxs * source_grid_freq_step + source_grid_center_freq

            #create the grid generator
            grid_mod = ExistsAtFrequencies(tone_freqs, interpret_as_absolute=True, error_on_mismatch=True)
            grid_gen = GridGenerator(None, grid_mod)

            #generate the signal grid
            sig_grid = Grid.generate("signal", grid_gen, on=with_grid)



        #pull out the signal data from the vst signal
        amplitudes = vst_sig.amplitudes
        if phase_in_deg:
            phases = np.deg2rad(vst_sig.phases)
        else: #not phase_in_deg
            phases = vst_sig.phases
        
        #build the signal data
        sig_data = amplitudes * (np.e)**(1j*phases)
        num_tones = int(sig_data.size)

        return cls(num_tones, sig_grid, power=float(power), 
                   power_type=power_type, with_tone_vals=sig_data, 
                   Z0=Z0, using_ref_grid=using_ref_grid, auto_level=auto_level)

    @classmethod
    def from_yaml(cls, 
                  file, 
                  with_grid:Grid,
                  signal_name:Union[str,None]=None, 
                  signal_idx:Union[int,None]=None,
                  power:float=1e-3, 
                  power_type:str="peak", 
                  Z0:float=50, 
                  using_ref_grid:Union[Grid,None]=None,
                  auto_level:bool=True):
        
        """Create a multitone signal from a yaml file."""
        #check if the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File does not exist: {file}")
                
        #load the yaml file
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        #get the signals property from the yaml file
        if "signals" not in data:
            raise RuntimeError(f"Provided yaml file does not contain 'signals' property: {file}")
        signals = data["signals"]

        #check if the signal name is provided
        if isinstance(signal_name, str):
            #find the signal with the specified name
            if signal_name not in signals:
                raise RuntimeError(f"Signal name '{signal_name}' not found in yaml file: {file}")
            data = signals[signal_name]
        elif isinstance(signal_idx, int):
            #find the signal with the specified index  
            if signal_idx < 0 or signal_idx >= len(signals):
                raise RuntimeError(f"Signal index {signal_idx} is out of bounds for yaml file: {file}")
            data = signals[signal_idx]
        else:
            #load the first signal in the yaml file
            if isinstance(signals, dict):
                data = signals
            else:
                raise RuntimeError(f"No signals found in yaml file: {file}")
        
        #now get the tones for the signal
        if "Tones" not in data:
            raise RuntimeError(f"Signal data does not contain 'Tones' property: {file}")
        tones = np.array(data["Tones"])

        #get the center frequency and frequency step
        if "Center Frequency" not in data or "Frequency Step" not in data:
            raise RuntimeError(f"Signal data does not contain 'Center Frequency' or 'Frequency Step' properties: {file}")
        center_freq = data["Center Frequency"]
        freq_step = data["Frequency Step"]

        #get the number of SSB tones
        num_ssb_tones = (tones.size - 1)/2
        if not num_ssb_tones.is_integer():
            raise RuntimeError(f"Number of tones in {file} is not even.")
        
        #build the tone indices
        tone_idxs = np.arange(-num_ssb_tones, num_ssb_tones + 1)
        #build the tone frequencies
        tone_freqs = center_freq + tone_idxs * freq_step

        #create the grid generator
        grid_mod = ExistsAtFrequencies(tone_freqs, interpret_as_absolute=True, error_on_mismatch=True)
        grid_gen = GridGenerator(None, grid_mod)
        #generate the signal grid
        sig_grid = Grid.generate("signal", grid_gen, on=using_ref_grid)
        
        #create the new signal
        return cls(tone_idxs.size, sig_grid, power=float(power), 
                   power_type=power_type, with_tone_vals=tones, 
                   Z0=Z0, using_ref_grid=using_ref_grid, auto_level=auto_level)

    @classmethod
    def null_signal(cls, with_grid:Grid):
        #create a signal on the specified grid where all absolute values are 0

        #initialize the grid to zero
        sig_data = np.full(with_grid.size, 0, dtype=np.dtype("complex"))
        num_tones = with_grid.size

        #return the class
        return cls(int(num_tones), with_grid, power=float(0), 
                   power_type="average", with_tone_vals=sig_data)
    
    # @classmethod
    # def load(cls, file, grid, grid_name:Union[str,None]):
    #     #Load a signal from an h5 file
    #     sig_def = SignalDefinition.load(file)

    #     #get the signal's grid
    #     if not grid_name is None:
    #         grid_src = StaticGridSource(sig_def.frequencies)
    #         grid_gen = GridGenerator(grid_src)
    #         signal_grid = Grid.generate(grid_name, using=grid_gen, on=grid)
    #     else: #the grid provided is the target grid
    #         signal_grid = grid

    #     #now build the signal
    #     return cls(sig_def.num_tones, 
    #                signal_grid, 
    #                power=sig_def.power,
    #                power_type=sig_def.power_type,
    #                with_tone_vals=sig_def.a0, 
    #                Z0=sig_def.z0,
    #                auto_level=sig_def.auto_level)

    #Class Validator Methods
    @staticmethod
    def __is_valid_grid(grid) -> bool:
        return isinstance(grid, Grid)
    
    @staticmethod
    def __is_valid_num_tones(num_tones:int) -> bool:
        #first check if the number of tones is the right type
        return isinstance(num_tones, int) and num_tones > 0

    @staticmethod
    def __is_valid_power(p:float) -> bool:
        #first check if the number of tones is the right type
        return isinstance(p, float) and (p >= 0)

    @staticmethod
    def __is_valid_power_type(pt:str) -> bool:
        return isinstance(pt,str) and (pt in ["average", "peak"])
    
    @staticmethod
    def __is_valid_tone_value_type(tv:np.ndarray) -> bool:
        return isinstance(tv, np.ndarray) and (tv.dtype is np.dtype("complex"))

    def __init__(self, num_tones:int, with_grid:Grid, power:float=1e-3, power_type:str="peak", 
                 with_tone_vals:Union[None, np.ndarray]=None, Z0:float=50, using_ref_grid:Union[Grid,None]=None,
                 auto_level:bool=True, td_max:Union[float,None]=None):
        
        #initialize hidden variables
        self.__power_type = None
        self.__power = None
        self.__a0 = None
        self.__auto_level = False #so we can set initial values
        self.__Z0 = None

        #some properties are only set during intialization
        assert MultitoneSignal.__is_valid_grid(with_grid), "Provided grid is not valid."
        assert MultitoneSignal.__is_valid_num_tones(num_tones), "Provided number of tones is not valid"
        assert with_grid.size == num_tones, "Number of tones must match the size of the grid."
        
        #set the grid and number of tones
        self.__grid = with_grid
        self.__num_tones = num_tones

        #set the power type and power
        self.power_type = power_type
        self.power = power

        #now generate the relative tone values
        if with_tone_vals is None:
            self.a0 = np.full(self.num_tones, 1, dtype="complex")
        else:
            self.a0 = with_tone_vals

        #finally, set the system characteristic impedance
        Z0 = float(Z0)
        if isinstance(Z0, float) and (Z0 > 0):
            self.__Z0 = Z0
        
        #set the reference grid
        if using_ref_grid is None:
            self.__fft_ref_grid = self.grid.root
        elif self.grid.__shares_root_with(using_ref_grid):
            self.__fft_ref_grid = using_ref_grid
        else:
            raise RuntimeError("Invalid reference grid provided. It must share "
            "the root grid with this signal's grid.")
        
        #set the autoleveling option
        self.auto_level = auto_level

        #initialize the value of td_max
        if td_max is None:
            self.td_max = 4/max(self.grid.offset_freqs)
        else:
            self.td_max = td_max

        if self.__auto_level:
            self.__offset_amplitudes_to_power_level()
    
    def complex_baseband_time_domain(self)->tuple[np.ndarray, np.ndarray]:
        #return the time domain representation of the signal

        """
        First step is to build the data for the ifft. Use the fft reference grid
        to generate the time domain signal
        """
        
        #note that this assumes the reference grid is uniform and contiguous
        #the reference grid is set during class initialization and cannot be set. 
        #It is recommended to leave this alone 
        ref_grid = self.__fft_ref_grid

        #cast the tone data to the reference grid
        data = self.grid.cast(self.__a0, ref_grid, 0, np.dtype("complex"))
        #get the frequencies about dc
        freqs = ref_grid.offset_freqs

        #shift the data by re-indexing about the center of the grid (trust me, it works)
        #we may run into problems here if the reference grid is not odd. 
        data = data[ref_grid.index(about_center=True)]

        #the frequency step
        df = np.abs(freqs[1] - freqs[0])
        N = data.size
        dt = 1/(N*df)

        #TODO: add additional code to allow for oversampling here
        #this doesn't seem to be as simple as padding the signal with n 
        #extra points

        #calculate the padding factor to add 

        """
        Next, perform the ifft
        """
        #data is scaled by a factor of N
        baseband_vals= np.fft.ifft(data, N) * N
        time = np.arange(N) * dt

        return baseband_vals, time
    
    def __offset_amplitudes_to_power_level(self)->None:
        #this offsets the values of the signal to obtain a peak or average exchangeable power
        
        #calculate a signal scaling factor 
        if self.power == 0: 
            #if this is zero, than set the scaling factor to zero (and avoid the nasty fraction)
            power_scale_factor = 0
        elif self.__power_type == "average":
            #get the sum of the vector
            pres_avg_power = self.average_power
            #set the scale factor to apply
            power_scale_factor = self.power / pres_avg_power
        elif self.__power_type == "peak":
            #calculate the present peak power
            pres_peak_power = self.peak_power
            #set the scaling factor for the signal 
            power_scale_factor = self.power / pres_peak_power

        #update the signal 
        self.__a0 *= np.sqrt(power_scale_factor)
    
    #method for saving the signal 
    def save(self, file_path, index_to:Union[Grid,None]=None):
        
        # If the file exists, delete the file
        if os.path.exists(file_path):
            os.remove(file_path)

        #Get the signal indexing relative to some grid
        if index_to is None:
            index_to = self.grid.root
        
        #index the signal relative to the desired grid
        signal_indices = index_to.cast_index(self.grid, about_center=True)
        
        # Next we will create the hdf5 file
        with h5py.File(file_path,"w") as file:
            #create the groups of data that we will write the 
            grid_grp   = file.create_group("Reference_Grid")
            #now save the signal data to the file
            file.create_dataset("A0", data=self.a0)
            file.create_dataset("Frequencies", data=self.grid.freqs)
            file.create_dataset("Tone_Indices", data=signal_indices)
            file.attrs.create("Z0", self.Z0)
            file.attrs.create("Power", self.power)
            file.attrs.create("Power_Type", self.power_type)
            file.attrs.create("Auto_Level", self.auto_level)

            #save the reference grid data
            grid_grp.create_dataset("Frequencies", data=index_to.freqs)
            grid_grp.attrs.create("Frequency_Resolution", index_to.freqs)
            grid_grp.attrs.create("Name", index_to.name)



    #method for writing to data file for AWRDE and ADS
    def write_to_AWR_sig_file(self, file_path, min_power=-100):
        #write the excitation data file to the desired file path

        #get the start and stop indices
        root_grid = self.grid.root
        
        #get all the data cast to the root grid
        data = self.grid.cast(self.a0, root_grid, dtype="complex")

        #get the indices of the first and final nonzero values
        nzvs = np.nonzero(data)[0]
        nzvs_start_idx = nzvs[0]; nzvs_stop_idx = nzvs[-1]
        
        #only keep the values in the range
        data = data[np.arange(nzvs_start_idx,nzvs_stop_idx+1)]

        #calculate the powers in dBm
        power_data = 10*np.log10(np.abs(data)**2 * 1e3)
        zero_vals = np.isneginf(power_data)
        #set the maximum value to zero
        power_data = power_data - np.max(power_data)
        #now set any inf values to the minimum 
        power_data[zero_vals] = min_power
        
        #calculate the phases in degrees
        phase_data = np.angle(data,deg=True)
        phase_data[zero_vals] = 0


        #now get the frequency step 
        freq_step = root_grid.freqs[1] - root_grid.freqs[0]

        #now write the file
        with open(file_path, "w") as file:
            #start by writing the header data
            print("spectrum=DOUBLE_SIDED", file=file)
            print(f"freq = {freq_step:0.3f}", file=file)

            #now write the data
            for idx,pwr in enumerate(power_data):
                #get the nex power and phase
                phase = phase_data[idx]
                #print the power and 
                print(f"{pwr:0.9}\t{phase:0.9}", file=file)

    def write_to_ADS_sig_file(self, file_path):
        #write the data to a file that ADS can read

        #Unlike AWR, ADS uses time-domain data for the signal.
        a0t, time = self.complex_baseband_time_domain()

        #convert a0t to v0t as ADS works with voltage
        v0t = a0t * np.sqrt(self.__Z0)

        #some things to save in the headder of the file
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frequency_step = self.freqs[1] - self.freqs[0]
        time_step = time[1] - time[0]
        period = time[-1] - time[0]
        avg_power  = 10 * np.log10(np.abs(self.average_power * 1e3))
        peak_power = 10 * np.log10(np.abs(self.peak_power * 1e3))

        #build the reference impedance string
        z0_str = f"{np.real(self.__Z0):0.1f}"
        if np.sign(np.imag(self.__Z0)) == 1:
            z0_str = z0_str + f"+ {self.__Z0:0.1f}j"
        else:
            z0_str = z0_str + f"- {self.__Z0:0.1f}j"

        #generate the header file
        header =   f"# MULTITONE SIGNAL GENERATED ON: {current_datetime}\n" \
                   f"# FOR TARGET SIMULATOR: ADS \n" \
                   f"# Reference Impedance: {z0_str} Ohms\n" \
                   f"# PEAK POWER: {peak_power} dBm\n" \
                   f"# AVERAGE POWER: {avg_power} dBm\n" \
                   f"# NUMBER OF TONES: {self.num_tones}\n" \
                   f"# FREQUENCY STEP: {frequency_step} Hz\n" \
                   f"# TIME STEP: {time_step} seconds\n" \
                   f"# Period: {period} seconds\n" \
                   f"# FORMAT CODE: tri (time, real, imaginary)"
        
        #now save the data
        with open(file_path, "w") as file:
            #write the header 
            print(header, file=file)

            #now write out the data, line-by-line
            for idx, t in enumerate(time):
                #write the main body of the file
                print(f"{time[idx]}\t{np.real(v0t[idx])}\t{np.imag(v0t[idx])}", file=file)

    @property
    def average_power(self)->float:
        #The average power is the sum of the power in
        #the individual tones. No need to calculate this
        #in the time domain
        return np.sum(np.abs(self.__a0)**2 / 2)
    
    @property 
    def peak_power(self)->float:
        #Return the estimated peak power of the signal

        #first get the baseband signal in time domain
        a0t, _ = self.complex_baseband_time_domain()

        #return the peak power of the time domain waveform
        return np.max(np.abs(a0t)**2 / 2)
    
    @property
    def par(self)->float:
        #Return the peak to average power ratio of the current signal
        return self.peak_power / self.average_power

    @property
    def rel_a0(self)->np.ndarray:
        #Return the relative tone values wrt the current power type
        return self.__a0 / np.sqrt(self.__power)
    
    @property
    def freqs(self)->np.ndarray:
        #return the frequencies of the signal
        return self.grid.freqs

    @property
    def num_tones(self)->int:
        #return the number of tones
        return self.__num_tones

    @property
    def grid(self)->Grid:
        #return the grid
        return self.__grid

    @property
    def power_type(self)->str:
        #return the power type
        return self.__power_type

    @power_type.setter
    def power_type(self, new_type:str)->None:
        if MultitoneSignal.__is_valid_power_type(new_type):
            self.__power_type = new_type
        else:
            raise ValueError("Power type must be 'average' or 'peak'")

    @property
    def power(self)->float:
        #return the power type
        return self.__power

    @power.setter
    def power(self, new_power:float)->None:
        if MultitoneSignal.__is_valid_power(new_power):
            self.__power = new_power
        else:
            raise ValueError(f"Cannot set the power to: {new_power}")
        
        if self.auto_level:
            self.__offset_amplitudes_to_power_level()

    @property
    def a0(self)->np.ndarray:
        return self.__a0
    
    @a0.setter
    def a0(self,new_vals:np.ndarray)->None:
        if MultitoneSignal.__is_valid_tone_value_type(new_vals):
            if self.grid.is_compatible(new_vals):
                #set the tone values
                self.__a0 = new_vals
            else:
                raise RuntimeError(f"Tone values (size {new_vals.size}) are not compatible with the signal grid (size {self.grid.size})")
        else:
            raise TypeError("New tone values must be provided as a numpy array of complex values")
        
        if self.auto_level:
            self.__offset_amplitudes_to_power_level()
    
    @property
    def a0_normalized(self)->np.ndarray:
        #return the normalized tone values to the peak tone value
        return self.__a0 / np.max(np.abs(self.__a0))

    @property
    def v0(self)->np.ndarray:
        #Returns the tone values as standard power waves (what the vst bench measures)
        #a1 = (V+I*Z0)/2*sqrt(Z0)
        #v+ = (V+I*Z0)/2 (from Pozar)
        #so v+ = a1 * sqrt(Z0)
        return self.__a0 * np.sqrt(self.__Z0)
    
    @v0.setter
    def v0(self, new_vals:np.ndarray)->None:
        if MultitoneSignal.__is_valid_tone_value_type(new_vals):
            if self.grid.is_compatible(new_vals):
                #set the tone values
                self.__a0 = new_vals / np.sqrt(self.__Z0)
            else:
                raise RuntimeError(f"Tone values (size {new_vals.size}) are not compatible with the signal grid (size {self.grid.size})")
        else:
            raise TypeError("New tone values must be provided as a numpy array of complex values")
        
        if self.auto_level:
            self.__offset_amplitudes_to_power_level()

    @property
    def v0_normalized(self)->np.ndarray:
        #return the normalized values of v0
        return self.a0_normalized * np.sqrt(self.__Z0)

    @property
    def p0(self)->np.ndarray[complex]:
        #returns the real power of each tone 
        return np.abs(self.__a0)**2 / 2

    @property
    def p0_normalized(self)->np.ndarray:
        #returns the normalized power of each tone.
        #note that power is normally a0/2 however this is left out for 
        #relative power values
        return np.abs(self.a0_normalized)**2

    @property
    def auto_level(self)->bool:
        return self.__auto_level
    
    @auto_level.setter
    def auto_level(self, new_val:bool)->None:
        # if isinstance(new_val, bool):
        #     self.__auto_level = new_val
        self.__auto_level = bool(new_val)

    @property
    def Z0(self)->float:
        return self.__Z0

    @property
    def definition(self)->SignalDefinition:
        return SignalDefinition(self.freqs, self.a0, 
                                power=self.__power,
                                power_type = self.__power_type,
                                auto_level=self.__auto_level,
                                z0=self.__Z0)
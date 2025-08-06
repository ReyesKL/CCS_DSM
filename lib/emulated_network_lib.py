import matplotlib.pyplot as plt
import numpy as np

#import library components
# from VST_Measurement_System import MeasurementSystem as VST_Sys
from lib.active_tuner_grid_lib import Grid
from typing import Union
import warnings
from lib.multitone_signal_lib import MultitoneSignal
from  lib.VST_Active_Tuner import VST_Active_Tuner, LinkedGammaDebugPlotter
from lib.vst_util_lib import uiputfile
import skrf 


class EmulatedNetwork():
    """
    Class for network emulation and is an abstraction of the VST active tuner class.
    Originally this was going to be included inside of the tuner class, however; it 
    became apparent that this class would be more effective as a high-level interface
    object on it's own. It provides a wrapper method on the tuner's "move_to" 
    method and provides infrastructure to de-embedd the voltage waves out of the tuner.
    """
    #static and class methods
    @classmethod
    def thru(cls, tuner:VST_Active_Tuner, signal, measurement_grid:Union[Grid,None]=None):
        #Initialize the object as a perfect, matched, thru

        #get the measurment grid
        if measurement_grid is None:
            measurement_grid = tuner.grid.root

        #initialize the data array
        data = np.full((measurement_grid.size,2,2), 0, dtype=np.dtype('complex'))

        #set the individual s-parameters
        data[:,0,0] = complex(0) #s11
        data[:,0,1] = complex(1) #s12
        data[:,1,0] = complex(1) #s21
        data[:,1,1] = complex(0) #s22

        #now create the object
        return cls(tuner, signal, data, measurement_grid=measurement_grid, Z0=50, sp_data_source="python")

    #methods for validation and verification
    def __is_valid_on_tuner_grid(self, newVal:np.ndarray)->bool:
        return self.__measurement_grid.is_compatible(newVal)

    #constructors/deconstructors
    def __init__(self, 
                 tuner:VST_Active_Tuner, 
                 signal:MultitoneSignal, 
                 sp_data:np.ndarray,
                 measurement_grid:Union[Grid,None]=None, 
                 Z0:float = 50.0, 
                 sp_data_source:str="python",
                 port_on_dut:int=2,
                 deembed:bool=False):
        
        #Initialize the hidden properties
        self.__tuner            = None
        self.__signal           = None
        self.__network          = None
        self.__Z0               = None
        self.__port_on_dut      = None
        self.__deembed          = None

        #set the tuner
        if isinstance(tuner, VST_Active_Tuner):
            self.__tuner = tuner

        #set the signal 
        if isinstance(signal, MultitoneSignal):
            self.__signal = signal
        
        #if the measurement grid is not specifically specified assume
        #it is the root grid
        if measurement_grid is None:
            self.__measurement_grid = self.__tuner.Receiver.grid
        else:
            self.__measurement_grid = measurement_grid

        #set the characteristic impedance
        self.Z0 = Z0

        #set the de-embbed parameter
        self.deembed = deembed

        #set the port parameter
        self.port_on_dut = port_on_dut

        #setup the network
        self.set_network(sp_data, z0=self.Z0, data_source=sp_data_source)
    #     #TODO: Warn if the network is going active here.

    def import_spdata(self, sp_data, format_type:str='python'):
        #Matlab and python handle the s-parameter data in different ways
        
        #make sure the data is converted to a ndarray
        sp_data = np.array(sp_data).astype('complex')

        #for both cases the length of the spdata shape should be a tuple of three elements
        data_shape = sp_data.shape
        assert len(data_shape) == 3, "S-parameter data should always be provided in three dimensions"
        
        #there are two cases supported by this class 'matlab' and 'python'
        if format_type.lower() == 'matlab':
            #the data shape should be 2x2xN
            assert (data_shape[0]==2), 'Number of rows for a matlab s-parameter array should be 2'
            assert (data_shape[1]==2), 'Number of columns for a matlab s-parameter array should be 2'
            assert (data_shape[2]>=2), 'Number of data points must be greatter than or equal to 2'

            #now shift the dimensions so that we have a Nx2x2
            sp_data = np.moveaxis(sp_data, -1, 0)

        elif format_type.lower() == 'python':
            #the data shape should be 2x2xN
            assert (data_shape[1]==2), 'Number of rows for a matlab s-parameter array should be 2'
            assert (data_shape[2]==2), 'Number of columns for a matlab s-parameter array should be 2'
            assert (data_shape[0]>=2), 'Number of data points must be greatter than or equal to 2'

            #no need to shift the dimensions of the data since we are importing
        else:
            raise ValueError(f"Unrecognized format type option {format_type}")
        
        return sp_data
    
    def export_spdata(self, format_type:str="python"):
        #Prepare data for exporting to python or matlab

        #get the s-parameter data from the network
        sp_data = self.__network.s

        #now handle the shifting based on the format type
        if format_type.lower() == "matlab":
            #go from fxnxn to nxnxf
            sp_data = np.moveaxis(sp_data, 0, -1)
        elif format_type.lower() == "python":
            pass
        else:
            raise ValueError(f"Unrecognized format type {format_type}")
        
        #return the re-formatted data
        return sp_data
    
    def set_network(self, sp_data, z0:Union[float, np.ndarray[complex]]=50, frequency=None, data_source:str="matlab"):
        """
        Update the network definition for this network emulation. 
        """
        #handle the optional frequency argument
        if frequency is None:
            frequency = self.freqs
        else:
            #convert object to an np array
            frequency = np.array(frequency).astype('float')

        #make sure that the start and stop frequencies match that of this object
        if not (frequency[0] == self.freqs[0]):
            raise ValueError(f"Start frequency provided {frequency[0]:.3f}Hz does not match that of the network {self.freqs[0]:.3f}Hz")
                
        if not(frequency[-1] == self.freqs[-1]):
            raise ValueError(f"Stop frequency provided {frequency[-1]:.3f}Hz does not match that of the network {self.freqs[-1]:.3f}Hz")

        #import the sp-data
        sp_data = self.import_spdata(sp_data, format_type=data_source)

        #make sure that the final dimension matches the frequency dimension
        if not(sp_data.shape[0] == frequency.size):
            raise IndexError(f"Length of frequency vector {frequency.size} does not match the third dimension size of the s-parameter data {sp_data.shape[0]}")
        
        #build the network object
        frequency = skrf.frequency.Frequency.from_f(frequency, unit='Hz')
        self.__network = skrf.Network(frequency=frequency, s=sp_data, name='Emulator Network')

        #now re-interpolate the network object
        self.__network = self.__network.interpolate(self.freqs, basis='s', coords='cart')

        #update the tuner from the network
        self.update_tuner()

    def update_tuner(self):
        #Updates the desired reflection coefficient for the tuner

        """
        Update the tuner target reflection coefficient
        """

        if self.__port_on_dut == 1:
            #the desired reflection coefficient will be S22
            gamma_desired = self.__measurement_grid.cast(self.S11, self.__tuner.grid)
        elif self.__port_on_dut == 2:
            #the desired reflection coefficient will be S22
            gamma_desired = self.__measurement_grid.cast(self.S22, self.__tuner.grid)
        else:
            raise RuntimeError("Port on dut must be 1 or 2.")


        #set the desired reflection coefficient in the tuner
        self.__tuner.target_gamma = gamma_desired

        """
        Update the tuner signal data
        """
        #push the S0 data to the tuner
        self.__tuner.S0 = self.S0

    def move_to(self, iterations:int=10, error_threshold:float=-20.0, debug:bool=False)->tuple[bool, float, int]:
        #move to the current network

        #have the tuner move to the desired response
        success, final_err, ios = self.__tuner.move_to(iterations=iterations, 
                                                       error_threshold=error_threshold, 
                                                       debug=debug)
        
        #return the final values
        return success, final_err, ios

    def get_grid_by_name(self, grid_name:str, raise_error_if_not_found:bool=False):
        #Return the grid by a reference name. Names include:
        # "measurement", "rx", "receiver" - for the measurement grid
        # ""

        #makes sure that the grid name is a string
        if not isinstance(grid_name, str):
            raise TypeError("Grid name must be a string.")
        else:
            grid_name = grid_name.lower()

        #initialize the return variable
        grid = None

        #search through the network grids
        if grid_name == "signal":
            #it's important that this is handled at the network emulator level
            #as the tuner's signal will be a simple np.ndarray.
            grid = self.__signal.grid
        
        if grid is None:
            #try requesting the grid from the tuner
            grid = self.__tuner.get_grid_by_name(grid_name)

        #Method wrap-up.
        if (grid is None) and raise_error_if_not_found:
            raise TypeError(f"Could not find grid with name {grid_name}")
        else:
            #otherwise get the grid from the tuner
             return grid
        
    def get_abwaves(self, deembed:bool=None, on_grid:str=None)->tuple[np.ndarray, np.ndarray]:
        #Returns the de-embedded vp/vm waves for the entire tuner excitation of the network

        #determine the de-embedding state of this action.
        if deembed is None:
            #pull this from the deembedding state of this object
            deembed = self.deembed

        #pull in the s-parameters
        S = self.S
        s11 = S[:,0,0]; s12 = S[:,0,1]  
        s21 = S[:,1,0]; s22 = S[:,1,1]

        #this data exists on the measurement grid. If a grid has been
        #requested other than the measurement grid, let's cast it.
        if not (on_grid is None):
            #get the grid to cast to 
            grid = self.get_grid_by_name(on_grid)
        else:
            #default grid is the measurement grid
            grid = self.__measurement_grid

        #now set the a and b waves from the tuner's receiver
        #note that these may not be properly aligned anymore 
        #but this shouldn't be a problem here.
        A, B, freqs = self.__tuner.get_abwaves(on_grid=grid)

        #cast the s-parameters
        s11 = self.__measurement_grid.cast(s11, grid); s12 = self.__measurement_grid.cast(s12, grid)
        s21 = self.__measurement_grid.cast(s21, grid); s22 = self.__measurement_grid.cast(s22, grid)

        #finally determine the de-embedded a and b waves 
        if deembed:
            detS = s11 * s22 - s21 * s12
            if self.port_on_dut == 1:
                a = A * (  1 / s12) - B * (s11 / s12)
                b = A * (s22 / s12) - B * (detS / s12)
            elif self.port_on_dut == 2:
                a = A * (  1 / s21) - B * (s22  / s21)
                b = A * (s11 / s21) - B * (detS / s21)
            else:
                raise ValueError(f"Improper value for port index: {self.port_on_dut}")
        else:
            #return the a and b waves from the network's perspective
            a = B; b = A

        #return the final a/b waves
        return a, b, freqs

    def save(self, file_path:Union[str,None]=None, grid_name:Union[None,str]=None, signal_as_v0:bool=True):
        """
        Save the results from the emulated network
        """
        #Step 1: If the file path is not provided have the user provide it through the UI
        if file_path is None:
                file_path = uiputfile(f"Select destination for tuner data.", filetypes=[("Numpy File", "*.npz")])

        #Step 2: Handle the grid name
        if grid_name is None:
            grid_name = "tuner"

        #Step 3: Get the a/b waves
        #handle the two cases
        port_2_on_dut = self.port_on_dut==2
        #get the a/b waves
        a1, b1, freqs = self.get_abwaves(deembed=    port_2_on_dut, on_grid=grid_name)
        a2, b2, _     = self.get_abwaves(deembed=not port_2_on_dut, on_grid=grid_name)   

        #get the signal at the generator plane
        if signal_as_v0:
            S0 = self.__signal.grid.cast(self.__signal.v0, self.__tuner.grid)
        else:
            S0 = self.__signal.grid.cast(self.__signal.a0, self.__tuner.grid)

        #save the data
        np.savez(file_path,
                 freqs=freqs, 
                 S0=S0,
                 a1=a1, a2=a2, 
                 b1=b1, b2=b2, 
                 port_on_dut=self.port_on_dut)
        
    @property
    def S0(self)->np.ndarray:
        #get the signal with source excitation compensated by the network
        
        #get the signal excitation
        s0 = self.__signal.grid.cast(self.__signal.v0, self.__tuner.grid)

        #handle the network port cases
        if self.__port_on_dut == 1:
            #get the s-parameters to update the signal with
            s12 = self.__measurement_grid.cast(self.S12, self.__tuner.grid)
            s11 = self.__measurement_grid.cast(self.S11, self.__tuner.grid)
            #now apply the update to the signal 
            x = s12 / (1 - s11)
        elif self.__port_on_dut == 2:
            #get the s-parameters to update the signal with
            s21 = self.__measurement_grid.cast(self.S21, self.__tuner.grid)
            s22 = self.__measurement_grid.cast(self.S22, self.__tuner.grid)
            #now apply the update to the signal 
            x = s21 / (1 - s22)
        else:
            #this should never happen, but raise this just in case for trouble shooting purposes
            raise RuntimeError("Network port index is not valid. Must be 1 or 2")

        #make sure that 
        #assert np.all(np.abs(x) <= 1.01), f"Values of x found which surpasses 1.01, {np.max(np.abs(x)):.2f}. Please make sure that the s-parameters are passive"
        #assert np.all(np.abs(x) <= 1.5), f"Values of x found which surpasses 1.01, {np.max(np.abs(x)):.2f}. Please make sure that the s-parameters are passive"

        #get the equivalent excitation value
        s0 *= x
        
        #return the final value 
        return s0
    
    @property
    def S11(self)->np.ndarray:
        return self.__network.s[:,0,0]
    
    @property
    def S12(self)->np.ndarray:
        return self.__network.s[:,0,1]
    
    @property
    def S21(self)->np.ndarray:
        return self.__network.s[:,1,0]
    
    @property
    def S22(self)->np.ndarray:
        return self.__network.s[:,1,1]
    
    @property
    def S(self)->np.ndarray[complex]:
        return self.__network.s

    @property
    def freqs(self)->np.ndarray:
        #Return the frequencies associated with the measurement grid
        return self.__measurement_grid.freqs

    @property
    def tuner_freqs(self)->np.ndarray:
        #Return the frequencies associated with the excitation, i.e. tuner, grid
        return self.__tuner.grid.freqs
    
    @property
    def signal_freqs(self)->np.ndarray:
        #Returns the frequencies of the signal
        return self.__signal.freqs

    @property
    def Z0(self)->float:
        return self.__Z0
    
    @Z0.setter
    def Z0(self, new_val:float):
        if isinstance(new_val, (int, float)):
            self.__Z0 = float(new_val)
        else:
            raise TypeError(f"Unsupported type, {type(new_val).__name__},for Z0. Expecting float, or int.")

    @property
    def tuner(self):
        return self.__tuner
    
    @property
    def port_on_dut(self)->int:
        #Return the port index on the DUT
        return self.__port_on_dut
    
    @port_on_dut.setter
    def port_on_dut(self, new_val:int)->None:
        #Set the port index on the DUT
        new_val = int(new_val)
        
        #set the value
        if new_val in (1,2):
            self.__port_on_dut = new_val
        else:
            raise ValueError(f"Port index {new_val} not valid. Must be 1 or 2.")
        
    @property
    def deembed(self)->bool:
        #return whether de-embedding state is on
        return self.__deembed
    
    @deembed.setter
    def deembed(self, new_val:bool)->None:
        #Set the de-embedding state

        try:
            if not isinstance(new_val, (bool,float,int)):
                #make sure the user is aware of the irregularity in input
                warnings.warn("Value for de-embed state is an irregular typecast to bool.")
            #try to set the value
            new_val = bool(new_val)
        except:
            raise RuntimeError(f"Failure to cast to bool.")

        #set the value
        self.__deembed = new_val

class GeneralizedEmulatedNetwork():
    """
    This is a generalized and improved version of the emulated network class. 
    Class for network emulation and is an abstraction of the VST active tuner class.
    Originally this was going to be included inside of the tuner class, however; it 
    became apparent that this class would be more effective as a high-level interface
    object on it's own. It provides a wrapper method on the tuner's "move_to" 
    method and provides infrastructure to de-embedd the voltage waves out of the tuner.
    """

    def __init__(self, 
                 measurement_grid,              #the measurement grid of the network
                 network_data,                  #s-parameters of the network (required for initial call)
                 frequency,                     #frequency data for the network (required)
                 z0=50.0,                       #impedance data for the network (defaults to 50 Ohms)
                 data_source='python',          #the data source for the network data
                 enable_port_coupling = True,   #enable port coupling 
                 logger = None,                 #logger
                 verbose = True):               #set the verbosity level

        #for the network object
        self.__network                  = None     

        #Initialize the hidden properties
        self.__tuners                   = np.ndarray((0,),dtype="object")
        self.__signals                  = np.ndarray((0,),dtype="object")

        #assigments for each port
        self.__internal_ports           = np.ndarray((0,),dtype='int')
        self.__external_ports           = np.ndarray((0,),dtype='int')
        self.__unassigned_ports         = np.ndarray((0,),dtype='int')

        #other settings 
        self.__port_coupling_enabled    = enable_port_coupling

        #initialize the measurement grid
        self.__measurement_grid         = measurement_grid

        #setup the logger
        self.__logger                   = logger
        
        #setup the verbosity
        self.__verbose                  = None
        self.verbose                    = verbose

        #initialize the network
        self.set_network(network_data, frequency, z0=z0, data_source=data_source)
    
    def log(self,msg:str,msg_type:Union[str,None]=None):
        #only print if verbose is true
        if self.verbose:
            #make sure the message type is correct
            assert msg_type in [None, "info", "error", "warning"], "Invalid value for log message type"

            #handle the logger argument
            if self.__logger is None:
                print(msg)
            elif (msg_type is None) or (msg_type == "info"):
                self.__logger.info(msg)
            elif msg_type == "warning":
                self.__logger.warn(msg)
            elif msg_type == "error":
                self.__logger.error(msg)  


    def emulate(self, 
                max_network_iterations:int      = 3,
                network_error_threshold:float   = -20,
                init_with_coupling:bool         = False,
                max_A0_iterations:int           = 3,
                A0_error_threshold:float        = -20,
                max_gamma_iterations:int        = 3, 
                gamma_error_threshold:float     = -20,
                tuner_debug:bool                = False):
                # sim_net=None
        

        #Initialize the return values
        success=False
        ios=max_gamma_iterations
        pres_err_power = None

        #Update the tuner reflection coefficients and use coupling for first 
        #iteration if requested
        self.update_tuner(target_gamma=True, 
                          excitation=True,
                          include_internal_port_coupling=init_with_coupling)

        #Loop through each network update iteration 
        for network_iter in np.arange(max_network_iterations): 
            
            #Update the tuner excitations if the present iteration is greatter than zero
            if network_iter > 0: 
                #Initialize without internal port coupling on the first iteration (if requested)
                self.update_tuner(target_gamma=False, 
                                  excitation=True)
            
            #Move the tuners to the desired reflection coefficients
            self._move_tuners(max_A0_iterations=max_A0_iterations, 
                              A0_error_threshold=A0_error_threshold,
                              max_gamma_iterations=max_gamma_iterations, 
                              gamma_error_threshold=gamma_error_threshold,
                              debug=tuner_debug)
            
            #Print the current error
            pres_err_power = self.err()
            pres_err_power = 10*np.log10(pres_err_power*1e3)
            # print(f" Current emulator error is: {pres_err_power:.2f}")

            # #TEST: compare the current s-parameters against the simulated network
            # if sim_net is not None:
            #     #get the a/b waves
            #     tuner_grid = self.get_grid_by_name("tuner")
            #     a, b, f, _, _, _ = self.get_ab_waves(on_grid=tuner_grid, keep_measured_b_waves=True)
            #     active_tones = self.get_active_tones(on_grid=tuner_grid)

            #     #get the a and b waves
            #     a1 = a[active_tones,0,0]; b1 = b[active_tones,0,0]; b3 = b[active_tones,2,0]
            #     f = f[active_tones]

            #     #re-interpolate the s-parameters to the current frequencies
            #     sim_net = sim_net.interpolate(f)

            #     #now calculate the relative error in gain
            #     s11_meas = b1 / a1; s11_sim = np.squeeze(sim_net.s[:,0,0])
            #     s21_meas = b3 / a1; s21_sim = np.squeeze(sim_net.s[:,1,0])

            #     #compute the error in gain
            #     reflection_error = 10*np.log10(np.sqrt(np.mean(np.abs((s11_meas - s11_sim) / (s11_sim))**2)))
            #     gain_error = 10*np.log10(np.sqrt(np.mean(np.abs((s21_meas - s21_sim) / (s21_sim))**2)))

            #     #print the current gain error
            #     self.log(f"Present Reflection Error is: {reflection_error:0.1f}dB")
            #     self.log(f"Present Gain Error is: {gain_error:0.1f}dB")

            if pres_err_power <= network_error_threshold:
                self.log(f"EMULATION CONVERGED! Final error: {pres_err_power:.2f}")
                success = True
                ios = network_iter
                break
            else:
                self.log(f"Current emulation error: {pres_err_power:.2f}")
        
        return success, ios, pres_err_power
    

    def err(self):
        """
        Return the final network error. This is the total absolute error in the power at the 
        internal port plane of the network.
        """

        #get the tuner grid
        tuner_grid = self.get_grid_by_name('tuner_grid')

        #get the active tones 
        active_tones = self.get_active_tones(on_grid=tuner_grid)

        #Get the measured and simulated b-waves
        a, bs, _, ipm, epm, _ = self.get_ab_waves(on_grid=tuner_grid,keep_measured_b_waves=False)
        _, bm, _,   _,   _, _ = self.get_ab_waves(on_grid=tuner_grid,keep_measured_b_waves=True)

        #Get the difference in all b-waves
        Pdelta = np.abs(bm[active_tones] - bs[active_tones])**2
        # Pdelta = np.abs(np.abs(bm)**2 - np.abs(bs)**2)
        Apower = np.abs(a[active_tones])**2
        
        # #Get the difference in the b-waves
        # Pdelta = np.abs(bm[ipm.astype('bool')] - bs[ipm.astype('bool')])**2 / (2*50)
        # Apower = np.abs(a[ipm.astype('bool')])**2 / (2*50)
        
        #The error is the ratio of the difference in B-waves and the power in all A-waves
        return np.sum(Pdelta) / np.sum(Apower)

        # #Get the differences in the powers in the b-waves
        # Pdelta = (np.abs(bm[ipm.astype('bool')])**2 - np.abs(bs[ipm.astype('bool')])**2) / 100

        # #Return the total difference in power
        # return np.sum(np.abs(Pdelta))
    

    
    def _move_tuners(self, 
                    max_A0_iterations:int       = 3,
                    A0_error_threshold:float    = -20,
                    max_gamma_iterations:int    = 3, 
                    gamma_error_threshold:float = -20, 
                    debug:bool                  = False):
        """
        Move to the network definition estabilished by this class. 
        """

        #handle special debugger types
        for tuner in self.__tuners:
            if isinstance(tuner.plotter, LinkedGammaDebugPlotter): 
                #configure the plotter if it is linked plotter for tracking the count
                tuner.plotter.set_tuner_iterations(0, max_A0_iterations)
                tuner.plotter.set_tuner_iterations(1, max_A0_iterations)
                #set the maximum number of outter tuner iterations
                tuner.plotter.total_iterations = max_gamma_iterations


        #Initialize the return variables
        success=False
        ios=max_gamma_iterations

        #Move each tuner value 
        for gamma_iter in np.arange(max_gamma_iterations):
            
            #Run each tuner through a source update
            for tuner in self.__tuners: 
                #move the tuner to the desired value
                tuner.move_to(iterations=max_A0_iterations, 
                              error_threshold=A0_error_threshold, 
                              debug=debug)

            #Collect the total reflection ceofficient error 
            gamma_error = np.zeros((self.__tuners.size,), dtype='complex')

            #update the a/b waves of each tuner and calculate the new error in gamma
            for idx, tuner in enumerate(self.__tuners): 
                #make update the signal states, i.e. a/b waves
                tuner.update()
                #calculate the new error in gamma and append to the present error
                gamma_error[idx] += tuner.gamma_error

            #now compute the rms value of the error in gamma 
            # gamma_error = np.sqrt(np.mean(np.abs(gamma_error)**2))
            gamma_error = np.sum(np.abs(gamma_error))

            #convert to dB
            gamma_error = 10*np.log10(gamma_error)

            #Print the current gamma error
            # print(f"  Current gamma error is: {gamma_error:.2f}")
            # self.log(f"  Current gamma error is: {gamma_error:.2f}")

            #handle the break condition 
            if gamma_error <= gamma_error_threshold:
                self.log(f"Tuners converged with final error: {gamma_error:.2f}")
                success = True
                ios = idx + 1
                break
            else:
               self.log(f"Current tuner error: {gamma_error:.2f}") 

        #Reset the debugger plots
        for tuner in self.__tuners: 
            if isinstance(tuner.plotter, LinkedGammaDebugPlotter): 
                tuner.plotter.reset_counts()
        
        #The update has completed return the final results
        return success, ios, gamma_error

    def set_network(self, data:Union[float, np.ndarray[complex]], frequency=None, z0=50.0, data_source:str="python"):
        """
        Set the new network definition. 
        """
        #generate the new s-parameter data and frequency
        sp_data, frequency = self.__import_spdata(data, frequency=frequency, format_type=data_source)

        #create the frequency object required for the network
        frequency = skrf.frequency.Frequency.from_f(frequency, unit='Hz')

        #make a new network object
        new_network = skrf.Network(frequency=frequency, s=sp_data, z0=z0, name='Emulator Network')

        #now re-interpolate the network object
        new_network = new_network.interpolate(self.freqs, basis='s', coords='cart')
        
        #update the port assignments
        unassigned_ports = np.arange(1, new_network.number_of_ports+1)

        #first remove ports that are out of range 
        internal_ports_to_keep = np.logical_and(self.__internal_ports >= 1,self.__internal_ports <= new_network.number_of_ports)
        external_ports_to_keep = np.logical_and(self.__external_ports >= 1,self.__external_ports <= new_network.number_of_ports)

        #update the internal ports
        self.__internal_ports   = self.__internal_ports[internal_ports_to_keep] 
        self.__tuners           = self.__tuners[internal_ports_to_keep]
        unassigned_ports        = unassigned_ports[np.logical_not(np.isin(unassigned_ports, self.__internal_ports))]

        #update the external ports
        self.__external_ports   = self.__external_ports[external_ports_to_keep]
        self.__signals          = self.__signals[external_ports_to_keep]
        unassigned_ports        = unassigned_ports[np.logical_not(np.isin(unassigned_ports, self.__external_ports))]

        #now set the unassigned ports
        self.__unassigned_ports = unassigned_ports

        #set the network
        self.__network = new_network 

    def __is_port_valid(self,port_idxs:int)->bool:
        #make sure the port index is an array of integers
        port_idxs = np.array(port_idxs,dtype="int")
        
        return (((np.unique(port_idxs).size) == port_idxs.size) and 
                np.all(port_idxs >= 1) and
                np.all(port_idxs <= self.number_of_ports))

    def unassign_port(self, target_ports:int):
        """
        Unassign the port from an internal or external port
        """
        
        #make sure that the target ports are set as a list or tuple
        if not isinstance(target_ports, (list,tuple,np.ndarray)):
            target_ports = [target_ports]

        #make sure the target ports are cast to an array of integers
        target_ports = np.array(target_ports, dtype='int')

        #remove from the external ports
        is_external_port = np.isin(target_ports, self.__external_ports)
        self.__rem_external_port(target_ports[is_external_port])
        self.__unassigned_ports = np.unique(np.concatenate((self.__unassigned_ports, target_ports[is_external_port])))

        #remove from the internal ports
        is_internal_port = np.isin(target_ports, self.__internal_ports)
        self.__rem_internal_port(target_ports[is_internal_port])
        self.__unassigned_ports = np.unique(np.concatenate((self.__unassigned_ports, target_ports[is_internal_port])))

    def assign_external_port(self, port_index:int, signal_obj):
        """
        Assigns the designated port as an external port and sets the corresponding signal object. 
        """
        
        #make sure that the port index is a list or tuple
        if not isinstance(port_index, (list,tuple)):
            port_index = [port_index]

        #set the port index to an integer
        port_index = np.array(port_index, dtype="int")
        
        #make sure the port index is in range
        if not self.__is_port_valid(port_index):
            raise TypeError(f"Provided port index {port_index} is not valid")
        elif not port_index.size == 1:
            raise TypeError(f"Only one external port can be assigned at a time")

        #first, unassign the port to keep things clean
        self.unassign_port(port_index)

        #now, add the port index to the stack
        self.__external_ports = np.concatenate((self.__external_ports, port_index))
        self.__signals  =  np.concatenate((self.__signals, np.array([signal_obj], dtype="object")))

        #keep all elements that are not the present port index
        self.__unassigned_ports = self.__unassigned_ports[np.logical_not(np.isin(self.__unassigned_ports,port_index))]

    def assign_internal_port(self, port_index:int, tuner_obj):
        """
        Assigns the designated port as an internal port and sets the corresponding tuner object. 
        """
        #make sure that the port index is a list or tuple
        if not isinstance(port_index, (list,tuple)):
            port_index = [port_index]

        #set the port index to an integer
        port_index = np.array(port_index,dtype="int")

        #make sure the port index is in range
        if not self.__is_port_valid(port_index):
            raise TypeError(f"Provided port index {port_index} is not valid")
        elif not port_index.size == 1:
            raise TypeError(f"Only one internal port can be assigned at a time")

        #first, unassign the port to keep things clean
        self.unassign_port(port_index)

        #now, add the port index to the stack
        self.__internal_ports = np.concatenate((self.__internal_ports, port_index))
        self.__tuners  =  np.concatenate((self.__tuners, np.array([tuner_obj], dtype="object")))

        #keep all elements that are not the present port index
        self.__unassigned_ports = self.__unassigned_ports[np.logical_not(np.isin(self.__unassigned_ports,port_index))]

    def __rem_external_port(self, target_ports:int):
        
        #get the external ports from the object
        external_ports = self.__external_ports
        
        #now determine which ports should be removed
        to_keep = np.logical_not(np.isin(external_ports, np.array(target_ports,dtype='int')))

        #now remove the ports and associated signals
        self.__external_ports = self.__external_ports[to_keep]
        self.__signals = self.__signals[to_keep]

    def __rem_internal_port(self, target_ports:int):
        #get the external ports from the object
        internal_ports = self.__internal_ports
        
        #now determine which ports should be removed
        to_keep = np.logical_not(np.isin(internal_ports, np.array(target_ports,dtype='int')))

        #now remove the ports and associated signals
        self.__internal_ports = self.__internal_ports[to_keep]
        self.__tuners = self.__tuners[to_keep]
        
    def __import_spdata(self, sp_data, frequency=None, format_type:str='python'):
        #Matlab and python handle the s-parameter data in different ways

        #handle the optional frequency argument
        if frequency is None:
            frequency = self.freqs
        else:
            #convert object to an np array
            frequency = np.array(frequency, dtype='float')

        #make sure that the start and stop frequencies match that of this object
        if not (frequency[0] <= self.freqs[0]): 
            raise ValueError(f"Start frequency provided {frequency[0]:.3f}Hz must be less than or equal to that of the network {self.freqs[0]:.3f}Hz")
            # raise ValueError(f"Start frequency provided {frequency[0]:.3f}Hz does not match that of the network {self.freqs[0]:.3f}Hz")
                
        if not(frequency[-1] >= self.freqs[-1]):
            raise ValueError(f"Stop frequency provided {frequency[-1]:.3f}Hz must be greatter than or equal to that of the network {self.freqs[-1]:.3f}Hz")
            # raise ValueError(f"Stop frequency provided {frequency[-1]:.3f}Hz does not match that of the network {self.freqs[-1]:.3f}Hz")

        #make sure the data is converted to a ndarray
        sp_data = np.array(sp_data, dtype='complex')

        #for both cases the length of the spdata shape should be a tuple of three elements
        data_shape = sp_data.shape
        assert len(data_shape) == 3, "S-parameter data should always be provided in three dimensions"
        
        #there are two cases supported by this class 'matlab' and 'python'
        if format_type.lower() == 'matlab':
            #the data shape should be 2x2xN
            assert (data_shape[0]>=2), 'Number of rows for a matlab s-parameter array should be 2'
            assert (data_shape[1]>=2), 'Number of columns for a matlab s-parameter array should be 2'
            assert (data_shape[0] == data_shape[1]), 'Number of rows and columns for matlab s-parameters should be equal.'
            assert (data_shape[2]>=2), 'Number of data points must be greatter than or equal to 2'

            #now shift the dimensions so that we have a Nx2x2
            sp_data = np.moveaxis(sp_data, -1, 0)

        elif format_type.lower() == 'python':
            #the data shape should be 2x2xN
            assert (data_shape[1]>=2), 'Number of rows for a python s-parameter array should be 2'
            assert (data_shape[2]>=2), 'Number of columns for a python s-parameter array should be 2'
            assert (data_shape[1] == data_shape[2]), 'Number of columns and 3rd axis elements must be equal for python s-parameter data.'
            assert (data_shape[0]>=2), 'Number of data points must be greatter than or equal to 2'

            #no need to shift the dimensions of the data since we are importing
        else:
            raise ValueError(f"Unrecognized format type option {format_type}")
        
        return sp_data, frequency
    
    def update_tuner(self, internal_port_idx:int=None, target_gamma=True, excitation=True, include_internal_port_coupling=True, )->None:
        """
        Push updates to the tuner at index tuner_idx. If none is provided, then update all tuners. 
        """
        
        #First, handle the internal port index arguments
        if internal_port_idx is None:
            internal_port_idx = np.arange(self.number_of_internal_ports, dtype='int') + 1
        else:
            #make sure it's a list if not already an array (avoid 0d errors)
            if not isinstance(internal_port_idx, (list, tuple)):
                internal_port_idx = [internal_port_idx]
            #now turn it into an numpy array
            internal_port_idx = np.array(internal_port_idx, dtype='int')
        
        #Next, we iterate through the internal port indices
        for port_idx in internal_port_idx:
            #0) Get the corresponding tuner to update
            tuner = self.__tuners[port_idx-1]

            #1) Update the reflection coefficient 
            if bool(target_gamma):
                tuner.target_gamma = self.__gamma_desired(port_idx)

            #2) Update the excitation 
            if bool(excitation): 
                #we always include the external port coupling 
                s0 = self.S0e(port_idx)
                #we include the internal port coupling upon request
                if include_internal_port_coupling: 
                    s0 += self.S0i(port_idx)
                #set the tuner signal excitation 
                tuner.S0 = s0

    def __gamma_desired(self, internal_port_idx:int)->np.ndarray[float]:
        """
        Get the corresponding reflection coefficient.
        """
        #Internal port indices start at one to stay consistent with the math
        internal_port_idx = int(internal_port_idx - 1)

        #get the corresponding network port and tuner
        n     = self.__internal_ports[internal_port_idx]
        tuner = self.__tuners[internal_port_idx]

        #return the corresponding reflection coefficient
        return self.__measurement_grid.cast(self.Snm(n,n), tuner.grid)

    def S0e(self, internal_port_idx:int)->np.ndarray[float]:
        """
        Return the external port excitations at the corresponding internal port index
        """

        #Part 0: Internal port indices start at one to stay consistent with the math
        internal_port_idx = int(internal_port_idx - 1)
        
        #Part 1: Get the tuner grid
        n     = self.__internal_ports[internal_port_idx]
        tuner = self.__tuners[internal_port_idx]
        Snn   = self.__measurement_grid.cast(self.Snm(n,n), tuner.grid)

        #Part 2: Calculate the external port excitations
        external_excitation = 0
        for idx in np.arange(self.number_of_external_ports):
            #get the index of the external port
            m = self.__external_ports[idx]
            #get the corresponding signal
            signal = self.__signals[idx]
            #now cast the current signal to the tuner grid and add it to the external excitation
            if signal is not None:
                #cast the signal to the grid of relevance
                s0 = signal.grid.cast(signal.v0, tuner.grid)
            else:
                #the signal should be a zero-like structure
                s0 = tuner.grid.zeros_like(dtype='complex')
            #get the transmission component of the s-parameter matrix
            Snm = self.__measurement_grid.cast(self.Snm(n,m), tuner.grid)
            #now calculate the external excitation contribution
            external_excitation += (s0 * Snm) 
        
        #return the external excitation at the corresponding port index
        return external_excitation / (1 - Snn)
    
    def S0i(self, internal_port_idx:int)->np.ndarray[float]:
        """
        Return the external port excitations at the corresponding internal port index
        """

        #Part 0: Internal port indices start at one to stay consistent with the math
        internal_port_idx = int(internal_port_idx - 1) 

        #Part 1: Get the tuner grid
        n     = self.__internal_ports[internal_port_idx]
        tuner = self.__tuners[internal_port_idx]

        #don't continue unless port coupling is enabled
        if not self.__port_coupling_enabled:
            #return 0 if port coupling is not enabled
            return 0
        else:
            #get the s-parameters
            Snn = self.__measurement_grid.cast(self.Snm(n,n), tuner.grid)

        #Part 2: Calculate the internal port excitation through the network 
        internal_excitation = 0
        for idx in np.arange(self.number_of_internal_ports):
            #get the index of the internal port
            m = self.__internal_ports[idx]
            #skip this iteration if m==n
            if n == m:
                continue
            else:
                #get the mth tuner
                tuner_m = self.__tuners[idx]
                #get the b-wave measured at port m
                bm = self.__measurement_grid.cast(tuner_m.Receiver.b, tuner.grid)
                #now update the internal excitation
                Snm = self.__measurement_grid.cast(self.Snm(n,m), tuner.grid)
            
            #now update the internal excitation
            internal_excitation += (bm * Snm)
        
        #return the external excitation at the corresponding port index
        return internal_excitation  / (1 - Snn)

    def Snm(self, to_prt:int, from_prt:int):
        #get the s-parameters 
        return self.__network.s[:, int(to_prt-1), int(from_prt-1)]
    
    def get_grid_by_name(self, grid_name:str, at_idx:Union[int,None]=None, raise_error_if_not_found:bool=False):
        #Return the grid by a reference name. Names include:
        # "measurement", "rx", "receiver" - for the measurement grid
        # ""

        #makes sure that the grid name is a string
        if not isinstance(grid_name, str):
            raise TypeError("Grid name must be a string.")
        else:
            grid_name = grid_name.lower()

        #initialize the return variable
        grid = None

        #search through the network grids
        if grid_name == "signal":
            #it's important that this is handled at the network emulator level
            #as the tuner's signal will be a simple np.ndarray.
            if at_idx is not None:
                grid = self.__signals[int(at_idx)].grid
            else:
                grid = self.__signals[0].grid
        
        if grid is None:
            #try requesting the grid from the tuner
            if at_idx is not None:
                grid = self.__tuners[int(at_idx)].get_grid_by_name(grid_name)
            else:
                for tuner in self.__tuners:
                    #try to find the grid by name
                    grid = tuner.get_grid_by_name(grid_name)
                    #if the grid is found break from the loop
                    if grid is not None:
                        break
            
        #Method wrap-up.
        if (grid is None) and raise_error_if_not_found:
            raise TypeError(f"Could not find grid with name {grid_name}")
        else:
            #otherwise get the grid from the tuner
             return grid
        
    def get_ab_waves(self,on_grid:Union[str,Grid,None]=None,keep_measured_b_waves=True)->tuple[np.ndarray[complex],np.ndarray[complex], np.ndarray[float], np.ndarray[int], np.ndarray[int], np.ndarray[int]]:
        """
        Compute the full a/b waves of the network from the measured and boundary condition data. 
        The optional argument keep_measured_b_waves determines whether the computed b-waves at the internal ports 
        or the measured b-waves should be retained. The default value is true.
        """
        #Handle the type of grid to use
        #this data exists on the measurement grid. If a grid has been
        #requested other than the measurement grid, let's cast it.
        if isinstance(on_grid,str):
            #get the grid to cast to 
            grid = self.get_grid_by_name(on_grid)
        elif isinstance(on_grid, Grid):
            #do nothing and keep the grid.
            assert self.__measurement_grid.is_compatible(on_grid), "Provided grid is not compatible with the network's measurement grid"
            #if the assertion passes use the specified grid
            grid = on_grid
        else:
            #otherwise assume that we are using the measurement grid
            grid = self.__measurement_grid

        #Step 0: Setup for the number of ports and frequencies
        freqs = self.__measurement_grid.cast(self.freqs,grid)
        n_freqs = freqs.size; n_ports = self.number_of_ports
        a_waves = np.zeros((n_freqs, n_ports, 1), dtype='complex')
        b_waves = np.zeros((n_freqs, n_ports, 1), dtype='complex')
        
        #get the s-parameters for the current network
        temp_net = self.__network.interpolate(grid.freqs, basis='s', coords='cart')
        s = temp_net.s

        #Step 1: Collect the data required to solve the system
        for idx in np.arange(self.number_of_ports):
            #there are three cases in the port type
            # 1) it's unassigned so the corresponding a-wave should be zero
            # 2) it's an external port, so the corresponding a-wave is it's assigned signal
            # 3) it's an internal port, so the corresponding a-wave is the assigned tuner b-wave

            #get the port index
            port_idx = idx + 1

            #now handle the different cases
            #NOTE: Technically this is not correct to use the voltage-wave definition for 
            #the a-waves on the external ports as the s-parameters might not be normalized to 
            #50 Ohms. For now it should be fine though as long as all ports are real and the same impedance.
            if np.isin(port_idx, self.unassigned_ports):
                #set the a-waves 
                a = self.__measurement_grid.cast(self.__measurement_grid.zeros_like(dtype='complex'),grid)
            elif np.isin(port_idx, self.external_ports):
                #get the index of the target signal
                target_signal = np.isin(self.external_ports,port_idx)
                #get the corresponding signal 
                signal = self.__signals[target_signal]
                signal = signal[0]
                #now cast the signal to the measurement grid
                if signal is None:
                    a = grid.zeros_like(dtype='complex')
                else:
                    a = signal.grid.cast(signal.v0, grid)
            elif np.isin(port_idx, self.internal_ports):
                #get the index of the target tuner
                target_tuner = np.isin(self.internal_ports,port_idx)
                #get the corresponding tuner
                tuner = self.__tuners[target_tuner]
                tuner = tuner[0]
                #now get the corresponding a_waves
                a = self.__measurement_grid.cast(tuner.Receiver.b,grid)
            else:
                raise RuntimeError(f"Could not find port at index: {port_idx}")
            
            #set the corresponding entry in the a-waves
            a_waves[:,idx,0] = a[:]

        #next, calculate all of the b-waves
        b_waves = np.matmul(s,a_waves)

        #finally, update the b-waves to the measured values for all internal ports
        if keep_measured_b_waves:
            for idx, p_idx in enumerate(self.__internal_ports):
                b_waves[:,p_idx-1,0] = self.__measurement_grid.cast(self.__tuners[idx].Receiver.a,grid)
        
        #Step 2: We need to mark the different port types
        internal_port_marker    = np.zeros_like(a_waves, dtype='int')
        external_port_marker    = np.zeros_like(a_waves, dtype='int')
        unassigned_port_marker  = np.zeros_like(a_waves, dtype='int')
        
        #start with the unassigned ports
        for idx, p_idx in enumerate(self.__unassigned_ports):
            unassigned_port_marker[:,p_idx-1,0] = int(idx+1)
        
        #next set the external ports
        for idx, p_idx in enumerate(self.__external_ports):
            external_port_marker[:,p_idx-1,0] = int(idx+1)
        
        #finally set the internal port marker
        for idx, p_idx in enumerate(self.__internal_ports):
            internal_port_marker[:,p_idx-1,0] = int(idx+1)
        
        #Now return all the relevant values
        return a_waves, b_waves, freqs, internal_port_marker, external_port_marker, unassigned_port_marker

    def get_active_tones(self, on_grid:Union[str, Grid, None]=None):
        """
        get_active_tones
        Returns the active tones as a boolean array for all tuners on the specified grid

        """
        #Handle the type of grid to use
        #this data exists on the measurement grid. If a grid has been
        #requested other than the measurement grid, let's cast it.
        if isinstance(on_grid,str):
            #get the grid to cast to 
            grid = self.get_grid_by_name(on_grid)
        elif isinstance(on_grid, Grid):
            #do nothing and keep the grid.
            assert self.__measurement_grid.is_compatible(on_grid), "Provided grid is not compatible with the network's measurement grid"
            #if the assertion passes use the specified grid
            grid = on_grid
        else:
            #otherwise assume that we are using the measurement grid
            grid = self.__measurement_grid

        #now get the active tones for each tuner
        active_tuner_tones = np.full((grid.size, self.number_of_internal_ports), False, dtype="bool")

        #get the active tones of each tuner
        for idx, tuner in enumerate(self.__tuners):
            #call the active tones method for each tuner
            active_tuner_tones[:,idx] = tuner.get_active_tones(on_grid=grid)
        
        #now perform a logical and over all tones
        return np.prod(active_tuner_tones,axis=1).astype('bool')

    @property
    def port_coupling_enabled(self)->bool:
        #Returns the state of port coupling
        return self.__port_coupling_enabled
    
    @port_coupling_enabled.setter
    def port_coupling_enabled(self, new_val:bool)->None:
        #Set the port coupling of this object
        self.__port_coupling_enabled = bool(new_val)

    @property
    def S(self)->np.ndarray:
        #return the s-parameters of the network
        return self.__network.s

    @property
    def number_of_ports(self)->int:
        #return the number of ports 
        return self.__network.number_of_ports
    
    @property
    def number_of_external_ports(self)->int:
        #return the number of external ports
        return len(self.__external_ports)
    
    @property
    def external_ports(self)->int:
        #return the external ports 
        return self.__external_ports
    
    @external_ports.setter
    def external_ports(self, new_val):
        raise RuntimeError("External ports can only be set through the set_network method.")
    
    @property
    def number_of_internal_ports(self)->int:
        #return the number of internal ports
        return len(self.__internal_ports)
    
    @property
    def internal_ports(self)->int:
        #return the internal ports
        return self.__internal_ports
    
    @internal_ports.setter
    def internal_ports(self, new_val)->None:
        raise RuntimeError("Internal ports can only be set through the set_network method.")
    
    @property
    def number_of_unassigned_ports(self)->int:
        #return the number of unassigned ports
        return len(self.__unassigned_ports)
    
    @property
    def unassigned_ports(self)->int:
        return self.__unassigned_ports
    
    @unassigned_ports.setter
    def unassigned_ports(self, new_val)->None:
        raise RuntimeError("Unassigned ports cannot be unassigned this way.")

    @property
    def freqs(self)->np.ndarray:
        #Return the frequencies associated with the measurement grid
        return self.__measurement_grid.freqs

    @property
    def z0(self)->float:
        #get the port impedances from the network
        return self.__network.z0
    
    @z0.setter
    def z0(self,new_val)->None:
        #sets the new port impedance
        self.__network.renormalize(new_val)

    @property
    def signals(self):
        #return the array of signals to use
        return self.__signals
    
    @property
    def verbose(self)->bool:
        return self.__verbose
    
    @verbose.setter
    def verbose(self, new_val:bool):
        self.__verbose = bool(new_val)
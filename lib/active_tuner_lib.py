import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union
import lib.multitone_signal_lib as mts

class DebugPlotter:
    #Class Methods

    #Static Methods
    
    #Constructor Deconstructor Methods
    def __init__(self):
        
        #create a new axis
        self.ax = None
        self.tuners = []
    
    #Methods for overloading (these are required by the tuner)
    def on_entry(self, src)->None:
        #import the smithplot library
        import smithplot

        #turn on interactive mode
        plt.ion()

        #clear the figure
        plt.clf()

        #set the axis
        self.ax = plt.subplot(1, 1, 1, projection='smith')

    def update(self, src)->None:
        #get the data relevant to the default implementation
        
        #first, pull in the reflection coefficients to plot
        gamma_meas = src.get_tuner_gamma()
        gamma_des = src.target_gamma

        #clean up the axis
        self.ax.cla()

        #plot the new data
        self.ax.plot(gamma_des, marker="x", linestyle="none", markersize=10)
        self.ax.plot(gamma_meas, marker="o", linestyle="none")

        #draw the current plot
        plt.draw()

        #pause so the plot can be updated
        plt.pause(0.1)
    
    def on_exit(self, src)->None:
        #turn off interactive mode
        plt.ioff()

    def register(self, tuner_obj)->None:
        ## Register the tuner with the source 
        self.tuners = self.tuners + [tuner_obj]

    def remove(self, tuner_obj)->int:
        ## Remove the tuner from the registry (return the integer where found)

        #find the tuner and remove it
        for idx, tuner in enumerate(self.tuners):
            if tuner is tuner_obj:
                self.tuners.pop(idx)
                return idx

        #otherwise raise an error
        raise RuntimeError("Could not find the specified tuner for removal")
    
    def get_tuner_index(self, tuner_ref)->int:
        
        #initialize the return value
        tuner_idx = None

        #check the different types
        if isinstance(tuner_ref, (int,float)):
            #make sure it's an int
            tuner_ref = int(tuner_ref)
            #make sure it is a valid index
            if (tuner_ref < len(self.tuners)) and (tuner_ref >= 0):
                tuner_idx = tuner_ref
            else:
                raise IndexError("Tuner reference index is out of range")
        elif isinstance(tuner_ref, str):
            #search for the tuner name
            for idx, tnr in enumerate(self.tuners):
                if tnr.name == tuner_ref:
                    tuner_idx = idx
                    break
            #if nothing was found, raise an error
            if tuner_idx is None:
                raise ValueError(f"{tuner_ref} is not a valid tuner name.")
        elif isinstance(tuner_ref, ActiveTuner):
            #search for the tuner name
            for idx, tnr in enumerate(self.tuners):
                if tnr is tuner_ref:
                    tuner_idx = idx
                    break
            #if nothing was found, raise an error
            if tuner_idx is None:
                raise ValueError(f"provided tuner has not been registered.")
        else:
            raise NotImplementedError("Tuner referencing not implemented for this.")
        
        #return the tuner index that was found
        return tuner_idx

class SignalViabilityMask:
    """
    This is a supporting class for the active tuner. It provides two viability masks that will be used
    to remove signal content from consideration by the tuner. 
        1) Transmit Mask - Mask for the transmit signal in frequency domain. Should provide true values for tones
                           that can be realized by the VST's AC source
        2) Receiver Mask - Frequency-domain mask for the received signal. The purpose of this mask is to remove content from 
                           the received frequency domain signal that should not be considered in the final error calculation. 

    TODO: Add additional methods for handling A0 PAR. Not exactly sure how to handle that. 
    """ 

    @staticmethod
    def __is_valid_threshold(val)->bool:
        #Determine whether or not the threshold is valid
        return isinstance(val, (int, float))
    
    def __init__(self, tuner, enabled:bool=True, transmit_mask_enabled:bool=True, receive_mask_enabled:bool=True,
                  threshold:Union[float,None]=-80.0, use_tx_mask_in_rx_mask:bool=True, transmitter_threshold:float=-80, 
                  receiver_threshold:float=-80, keep_tones:bool=False):
        #Default constructor for the viability mask

        #the tuner must be an active tuner
        if isinstance(tuner, ActiveTuner):
            self._src = tuner
        else:
            raise TypeError("Tuner must be an active tuner object")

        #private properties
        self.__tx_thresh                = None
        self.__rx_thresh                = None
        self.__enabled                  = None
        self.__tx_mask_enabled          = None
        self.__rx_mask_enabled          = None
        self.__tx_mask                  = None
        self.__prev_tx_mask             = None

        self.__rx_mask                  = None
        self.__use_tx_mask_in_rx_mask   = None
        self.__tone_locations           = None
        self.__ref_sig                  = None
        self.__always_keep_tones        = None

        #set the provided properties
        self.enabled                    = enabled
        self.transmit_mask_enabled      = transmit_mask_enabled
        self.receive_mask_enabled       = receive_mask_enabled
        self.use_tx_mask_in_rx_mask     = use_tx_mask_in_rx_mask
        self.keep_tones                 = keep_tones

        #if the threshold has been set, set both the receiver and transmitters
        #to that threshold, otherwise set them individually
        if isinstance(threshold, (int, float)):
            #set the thresholds to the noise floor
            self.transmitter_threshold=threshold
            self.receiver_threshold=threshold
        else:
            self.transmitter_threshold=transmitter_threshold
            self.receiver_threshold=receiver_threshold

    #methods 
    def update_transmit_mask(self)->np.ndarray[bool]:
        
        #get the source reference
        src = self._src        
        A0 = src.A0_expected

        # for tone in tone_list
            # if meas_tone_pow > noise_floor
            #   ex_tone_list.append(tone)
        # return ex_tone_list


        #Return a full mask if not enabled
        if (not self.__enabled) or (not self.__tx_mask_enabled):
            #get the tone locations
            tone_locs = self.tone_locations
            #handle the case where the tone locations are none
            if tone_locs is None:
                raise RuntimeError("Tone locations have not been set. Cannot return a default mask.")
                # #get the tuner signal
                # S0 = src.S0
                # #update this object's transmit mask
                # self.__tx_mask = np.full_like(A0, False, dtype=np.dtype("bool"))
                # self.__tx_mask[np.abs(S0) > 0] = True
                # #update this objects 
                # return self.__tx_mask
            else:

                #use the tone locations
                self.__tx_mask = tone_locs
                #return the receive mask
                return self.__tx_mask
        
        #now generate the transmit mask
        P0 = np.abs(A0)**2 / (2 * 50)

        #convert to dBm
        # P0 = np.where(P0 <= 0, 1e-10, P0) # avoid divide by zero
        P0 = 10*np.log10(P0) + 30

        #generate the mask as all values of P0 greater than the
        #specified threshold
        self.__tx_mask = P0 > self.__tx_thresh 

        #make sure the mask includes tones in the signal
        if self.__always_keep_tones:
            if self.tone_locations is None:
                raise RuntimeError("Tone locations not set yet.")
            else:
                self.__tx_mask = np.logical_or(self.__tx_mask, self.tone_locations)
        if self.__prev_tx_mask is not None:
            self.__tx_mask = np.logical_or(self.__tx_mask, self.__prev_tx_mask)
        self.__prev_tx_mask = self.__tx_mask
        return self.__tx_mask
        
    def update_receive_mask(self)->np.ndarray[bool]:
        """
        update the receiver mask
        This implementation of the receive mask assumes that the
        reflected wave power must surpass some threshold to be valid for 
        active tuner control.
        """

        #get the source reference
        src = self._src        
        B = src.B


        #Return a full mask if not enabled
        if (not self.__enabled) or (not self.__rx_mask_enabled):
            #get the tone locations
            tone_locs = self.tone_locations
            
            if self.tone_locations is None:
                raise RuntimeError("Tone locations have not been set. Cannot return a default mask.")
                # #get the signal tones
                # S0 = src.S0 
                # #update this object's transmit mask
                # self.__rx_mask = np.full_like(B, False, dtype=np.dtype("bool"))
                # self.__rx_mask[np.abs(S0)>0] = True
                # #update this objects 
                # return self.__rx_mask
            else:
                #use the tone locations
                self.__rx_mask = tone_locs
                #return the receive mask
                return self.__rx_mask
            
        #now generate the transmit mask
        P0 = np.abs(B)**2 / (2 * 50)

        #convert to dBm
        P0 = 10*np.log10(P0) + 30

        #generate the mask as all values of P0 greater than the specified threshold
        self.__rx_mask = P0 > self.__rx_thresh

        #combine the masks
        if self.__use_tx_mask_in_rx_mask:
            # self.__rx_mask = np.logical_or(self.__rx_mask, self.__tx_mask)
            self.__rx_mask = np.logical_and(self.__rx_mask, self.__tx_mask)

        #if always keep tones is true
        if self.__always_keep_tones:
            if self.tone_locations is None:
                raise RuntimeError("Tone locations not set yet.")
            else:
                self.__rx_mask = np.logical_or(self.__rx_mask, self.tone_locations)
        # plt.ioff()
        # ax1 = plt.subplot(111, projection='rectilinear')
        # ax2 = ax1.twinx()
        # ax1.plot(P0, color="black")
        # ax2.stem(self.__rx_mask)
        # ax1.grid()
        # ax1.axhline(self.__rx_thresh, color="red", linestyle="--")
        # plt.show()
        #return the receiver mask
        return self.__rx_mask

    #setters and getters
    @property
    def enabled(self)->bool:
        #
        return self.__enabled
    
    @enabled.setter
    def enabled(self, new_val)->None:
        #Setter for the enabled property
        if isinstance(new_val, bool):
            self.__enabled = new_val
        else:
            raise TypeError("enabled must be a bool.")

    @property
    def transmit_mask_enabled(self)->bool:
        return self.__tx_mask_enabled
    
    @transmit_mask_enabled.setter
    def transmit_mask_enabled(self, new_val:bool)->bool:
        if isinstance(new_val, bool):
            self.__tx_mask_enabled = new_val
        else:
            raise TypeError("transmit_mask_enabled must be type bool")
    
    @property
    def receive_mask_enabled(self)->bool:
        return self.__rx_mask_enabled
    
    @receive_mask_enabled.setter
    def receive_mask_enabled(self, new_val:bool)->None:
        if isinstance(new_val, bool):
            self.__rx_mask_enabled = new_val
        else:
            raise TypeError("receive_mask_enabled must be type bool")

    @property
    def keep_tones(self):
        return self.__always_keep_tones
    
    @keep_tones.setter
    def keep_tones(self, new_val:bool):
        if isinstance(new_val, (bool,int,float)):
            self.__always_keep_tones = bool(new_val)
        else:
            raise TypeError("Keep tones must be boolean or readily convertable to boolean.")

    @property
    def tone_locations(self)->np.ndarray[bool]:
        if isinstance(self.__ref_sig, mts.MultitoneSignal):
            #get the reference signal
            sig  = self.__ref_sig
            #now return the signal cast as an array of booleans 
            return sig.grid.cast(sig.a0.astype(bool), self._src.grid, off_grid_vals=False, dtype=np.dtype("bool"))
        else:
            return self.__tone_locations
    
    @tone_locations.setter
    def tone_locations(self, new_val:Union[np.ndarray, mts.MultitoneSignal])->None:
        if isinstance(new_val, np.ndarray):
            #make sure that the reference signal is cleared
            self.__ref_sig = None
            #now set the array data mask
            self.__tone_locations = new_val.astype(bool)
        elif isinstance(new_val, mts.MultitoneSignal):
            #set the reference signal
            self.__ref_sig = new_val


    # @property
    # def transmit_mask(self)->np.ndarray:
    #     if (not self.__enabled) or (not self.__tx_mask_enabled):
    #         mask = np.full_like(self.__tx_mask, False)
    #         mask[np.abs(self._src.S0)>0] = True
    #         return mask
    #     else:
    #         return self.__tx_mask

    # @property
    # def receive_mask(self)->np.ndarray:
    #     if (not self.__enabled) or (not self.__rx_mask_enabled):
    #         mask = np.full_like(self.__tx_mask, False)
    #         mask[np.abs(self._src.S0)>0] = True
    #         return mask
    #     else:
    #         return self.__rx_mask

    @property
    def transmit_mask(self)->np.ndarray:
            return self.__tx_mask

    @property
    def receive_mask(self)->np.ndarray:
            return self.__rx_mask

    @property
    def use_tx_mask_in_rx_mask(self)->bool:
        return self.__use_tx_mask_in_rx_mask
    
    @use_tx_mask_in_rx_mask.setter
    def use_tx_mask_in_rx_mask(self, new_val:bool)->None:
        if isinstance(new_val, bool):
            self.__use_tx_mask_in_rx_mask = new_val
        else:
            raise TypeError("use_tx_mask_in_rx_mask must be a bool")

    @property
    def transmitter_threshold(self)->Union[float,None]:
        return self.__tx_thresh

    @transmitter_threshold.setter
    def transmitter_threshold(self, new_val:Union[int, float])->None:
        if SignalViabilityMask.__is_valid_threshold(new_val):
            self.__tx_thresh = float(new_val)
        else:
            raise TypeError("Transmitter threshold must be an integer or float.")
    
    @property
    def receiver_threshold(self)->Union[float,None]:
        #Returns the receiver threshold. 
        return self.__rx_thresh
    
    @receiver_threshold.setter
    def receiver_threshold(self, new_val:Union[int, float])->None:
        if SignalViabilityMask.__is_valid_threshold(new_val):
            self.__rx_thresh = float(new_val)
        else:
            raise TypeError("Receiver threshold must be an integer or float")

class ActiveTuner:
    """
    This is the abstract interface class for an wideband active tuner object. Some assumptions are made in the implementation of this class:
        1) All measurements are in frequency domain
        2) All measurements are aligned
        3) All measurements exist on an externally defined excitation grid
    Definitions of the significant properties are
        A               : aligned a-wave of the tuner on the excitation grid
        B               : aligned b-wave of the tuner on the excitation grid
        A0              : total measured tuner excitation 
        A0_expected     : the expected value of the A0 value     
        S0              : the set signal excitation 
        T0              : the set tuner excitation
        B0              : the dut response excitation
        gamma_0         : the tuner's known reflection coefficient
        gamma_in_0      : the dut's estimated reflection coefficient
    """
    def __init__(self, debug_plotter=None, signal_masker=None):
        #Initialize the active tuner object

        #methods private to this class
        self.__plotter = None
        self.__target_gamma = None

        #set the initialization state of the tuner
        self.is_initialized = False

        #set the debug plotter
        if isinstance(debug_plotter, DebugPlotter):
            self.plotter = debug_plotter
        else:
            self.plotter = DebugPlotter()

        #set the gamma viability judge element
        if isinstance(signal_masker, SignalViabilityMask):
            self.sig_mask = signal_masker
        else:
            self.sig_mask = SignalViabilityMask(self)

    def initialize(self):
        """
        Main initialization method (override by sub-class)
        This method should perform the following tasks:
            1) make sure that all required values and properties have been set
            2) perform any initialization procedures
            3) call the parent class initialization at the end
        """
        
        #Run tuner initialization 
        self.is_initialized = True

    def shutdown(self):
        """
        Main shutdown method (override by sub-class)
        This method should perform the following tasks:
            1) return any required properties to state None
            2) perform any shutdown tasks, e.g. shut of source
            3) call the parent class shutdown method
        """
        #at the very end, set this to false
        self.is_initialized = False

    def update_A0(self):
        #update the value of the measured tuner excitation
        A = self.A
        B = self.B
        gamma_0 = self.gamma_0

        #calculate the new value of a0
        self.A0 = (A - B * gamma_0)/(1 - gamma_0)

    def update_B0(self):
        #update the value of the dut's excitation
        A = self.A
        B = self.B
        gamma_in_0 = self.gamma_in_0

        #update the value of b0
        self.B0 = (B - A * gamma_in_0)/(1 - gamma_in_0)

    # def update_T0(self, gamma_des):

    #     #get the signal and dut response
    #     S0 = self.S0
    #     B0 = self.B0
        
    #     #get the reflection coefficients
    #     gamma_in_0 = self.gamma_in_0
    #     gamma_0 = self.gamma_0

    #     #update the T0 Value
    #     self.T0 = ((S0 * (gamma_in_0 - 1) * (gamma_des - gamma_0) + 
    #                 B0 * (1 - gamma_in_0) * (gamma_des * (1 - gamma_0 * gamma_in_0) - gamma_0 * (1 - gamma_des * gamma_in_0)))/
    #                ((1 - gamma_des * gamma_in_0) * (1 - gamma_0)))

    def update_T0(self):
        #the desired gamma is the target gamma
        gamma_des = self.target_gamma

        #get the signal and dut response
        S0 = self.S0
        B0 = self.B0
        
        #get the reflection coefficients
        gamma_in_0 = self.gamma_in_0
        gamma_0 = self.gamma_0

        #update the T0 Value
        self.T0 = ((S0 * (gamma_in_0 - 1) * (gamma_des - gamma_0) + 
                    B0 * (1 - gamma_in_0) * (gamma_des * (1 - gamma_0 * gamma_in_0) - gamma_0 * (1 - gamma_des * gamma_in_0)))/
                   ((1 - gamma_des * gamma_in_0) * (1 - gamma_0)))

    def get_tuner_gamma(self)->np.ndarray[complex]:
        #return the value of the emulated tuner reflection coefficient
        A=self.A; B=self.B; S0=self.S0
        return (A - S0) / (B - S0)  

    def move_to(self, gamma_des:np.ndarray=None, iterations=10, error_threshold=-20, debug:bool=False, dynamic:bool=False)->tuple[bool, float, int]:
        #Implementation of the move_to method for the class

        #make sure the tuner is initialized
        if not self.is_initialized:
            raise RuntimeError("Tuner is not initialized! Please run initialize method before attempting to set reflection coefficient.")

        #Initialize the values to return
        success = False
        present_error = 0
        present_iteration = iterations

        #update the target gamma if provided, otherwise pull it from the 
        if gamma_des is None:
            #load the stored target_gamma value
            gamma_des = self.target_gamma
        else:
            #save the current desired gamma as the target
            self.target_gamma = gamma_des

        #Initialize the debug plot 
        if debug: 
            self.__plotter.on_entry(self)

        #Step 1: refresh the signal
        self.update_signal_states()

        #enter the main loop
        for idx in range(0, iterations):
            #Step 2: Update the excitation
            self.update_excitation(update_T0=((idx==0) or dynamic))

            #Step 3: refresh the signal states
            self.update_signal_states()

            #Step 4: get the new error and determine if the exit criteria has been met
            #get the error in dB
            present_error = 10*np.log10(np.abs(self.gamma_error))

            #update the debug plot 
            if debug: 
                self.__plotter.update(self)

            #if the present error is less than the desired error break
            # break from the loop 
            if present_error <= error_threshold:
                success = True
                present_iteration = idx + 1
                break
        
        #run plotter cleanup method 
        if debug:
            self.__plotter.on_exit(self)
        
        #return the final values of the tuner
        return success, present_error, present_iteration   
    
    def update_signal_states(self, measure:bool=True):
        """
        Update Signal States:
        This method updates the signals from the host hardware and updates the high-level signals of this class
        """
        #Step 1: Pull in the new signals
        if measure:
            #have the host system re-measure
            self.measure()
        else:
            self.update()

        #Step 2: Update A0
        self.update_A0()
        
        #Step 3: Update B0
        self.update_B0()

    def update_excitation(self, update_T0:bool=True):
        """
        Update excitation
        This method calculates a new tuner excitation and applies it to the physical system
        """

        #Step 1: Update T0 from the measurement data
        if update_T0:
            self.update_T0()
        
        #Step 2: Generate a new transmit mask
        self.sig_mask.update_transmit_mask()

        #Step 3: Apply the updated excitation 
        self.apply_present_excitation()

    """
    Define Abstract Methods
    """
    @abstractmethod
    def _is_target_gamma_valid(self, new_gamma:np.ndarray)->bool:
        pass

    @abstractmethod
    def measure(self):
        """
        This method depends on the exact implementation of this class.
        This method should trigger a new A/B wave measurement. It is assumed
        that the the resulting A and B waves are pre-aligned when being brought 
        into the tuner. 
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        The alternative method of pulling signals into this class without performing 
        a measurement. The specific way this is implemented depends on the implementation 
        of this class. The result should be that the latest aligned A/B waves should be made 
        available to this class.
        """
        pass
    
    @abstractmethod
    def apply_present_excitation(self):
        """
        This method applies the present excitation of the active tuner to the current measurement 
        system. The exact form of this will be dependent on the implementation but the general
        form would be:
        
        #get the current excitation
        A0 = self.A0_expected
        
        #now apply it 
        self.apply(A0,self.sig_mask.transmit_mask)
        """
        pass

    @abstractmethod
    def apply(self, newVal):
        """
        Sets the new tuner excitation.
        """
        pass
    
    @abstractmethod
    def generate_source_update(self):
        """
        This generates an update vector for the transmitter. The exact way this
        is accomplished depends on the implementation. The suggested approach is:

        return self.A0_expected / self.A0
        """
        pass

    """
    Define the abstract properties
    """
    #for A
    @property 
    @abstractmethod
    def A(self):
        pass

    @A.setter
    def A(self, newVal):
        #A is set when the method "measure()" is called. It should
        #not be set manually.
        raise RuntimeError("Property 'A' cannot be set.")
    
    #for B
    @property 
    @abstractmethod
    def B(self):
        pass
    
    @B.setter
    def B(self, newVal):
        #B is set when the method "measure()" is called. It should
        #not be set manually.
        raise RuntimeError("Property 'B' cannot be set.")

    #for gamma_0
    @property 
    @abstractmethod
    def gamma_0(self):
        pass

    #for gamma_in
    @property 
    @abstractmethod
    def gamma_in_0(self):
        pass

    #For A0 
    @property 
    @abstractmethod
    def A0(self):
        pass

    @property
    def A0_expected(self)->np.ndarray:
        return self.S0 + self.T0

    #for B0
    @property 
    @abstractmethod
    def B0(self):
        #returns the excitation from the DUT
        pass

    #for S0 
    @property 
    @abstractmethod
    def S0(self):
        pass

    @S0.setter
    @abstractmethod
    def S0(self, newVal):
        #set the new signal on the current grid
        pass

    #for T0 
    @property 
    @abstractmethod
    def T0(self):
        pass
    
    @property
    def target_gamma(self)->np.ndarray[complex]:
        return self.__target_gamma
    
    @target_gamma.setter
    def target_gamma(self, new_target:np.ndarray[complex])->None:
        #a wrapper on the set target gamma method
        if self._is_target_gamma_valid(new_target): #this needs to be defined by the subclass implementation
            #update the target gamma
            self.__target_gamma = new_target
        else:
            #otherwise raise an error
            raise TypeError("Target Gamma provided is not valid.")

    @property
    def plotter(self)->DebugPlotter:
        #Plotter object should not be directly exposed
        return self.__plotter
    
    @plotter.setter
    def plotter(self,new_plotter):
        #update the plotter
        if isinstance(new_plotter,DebugPlotter):
            #assign the plotter
            self.__plotter = new_plotter
            #register the tuner with the plotter
            self.__plotter.register(self)
        else:
            raise TypeError("Plotter provided is not a valid debug plotter.")
    
    @property
    @abstractmethod
    def gamma_error(self)->float:
        #return the error (absolute) of the gamma method
        pass

    @property
    @abstractmethod
    def a0_error(self)->float:
        #return the error (absolute) of the current excitation
        pass
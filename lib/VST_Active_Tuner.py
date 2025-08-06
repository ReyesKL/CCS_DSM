#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

#for animation debugger
from PIL import Image
import matplotlib.animation as animation
import io

#import library components
from lib.active_tuner_lib import ActiveTuner
from lib.active_tuner_lib import DebugPlotter as tdp

# from VST_Measurement_System import MeasurementSystem as VST_Sys
from lib.active_tuner_grid_lib import Grid
from typing import Union
import warnings
from lib.multitone_signal_lib import MultitoneSignal
from lib.vst_util_lib import fullfile, fileparts, isfolder, db2mag, polar2complex, uiputfile

#TODO: Create a plotting interface for the VST_Active_Tuner. This should not be implemented in the tuner library.

#Debug plotters 
 

class AnimatedDebugPlotter():
    #methods for checking that the file path is correct
    @staticmethod
    def __is_valid_animation_file(file_name:str)->bool:
        if isinstance(file_name, str):
            #get the directory and extension of the path
            directory, _, extension = fileparts(file_name)
            if (((directory == "") or (isfolder(directory))) and
                (extension == ".gif")):
                return True
            else:
                return False
        else:
            raise TypeError("file name needs to be a string.")
    
    def __init__(self):
        
        #properties specific to this class
        self.__frames                       = []
        self.__animation_frame_duration     = float(1000)
        self.__is_recording                 = False

    def reset_animation(self):
        #reset the animation frames
        self.__frames = []

    def save_animation(self, filename:str):
        
        #make sure that the file name is valid
        if not AnimatedDebugPlotter.__is_valid_animation_file(filename):
            raise RuntimeError("Could not save animation. Filename was not valid.")

        #make sure that there's something to save
        assert len(self.__frames) > 1, "Nothing to save in the present animation."

        #now save the animation
        self.__frames[0].save(filename,
                                save_all=True,
                                append_images=self.__frames[1:],
                                duration=self.__animation_frame_duration,
                                loop=0)
        
    def update_animation(self, target_figure=None):
        #if we aren't recording return now
        if not self.__is_recording:
            return

        #if the target figure is provided set it to the current figure
        if target_figure is None:
            #use the current figure
            pass
        elif isinstance(target_figure, (int, float, str)):
            #the reference is a number or name
            plt.figure(target_figure)
        else: #assume it is a figure reference
            #assume it is a figure
            plt.figure(target_figure.number) 
        
        #get the buffer to send the image to
        buf = io.BytesIO()
        #save the figure to the buffer
        plt.savefig(buf, format="png")
        #return the pointer in the buffer to the starting point
        buf.seek(0)
        #append to the frames
        self.__frames.append(Image.open(buf))

    #properties
    @property
    def is_recording(self)->bool:
        self.__is_recording

    @is_recording.setter
    def is_recording(self, new_val:bool)->None:
        if isinstance(new_val, (bool, float, int)):
            self.__is_recording = bool(new_val)
        else:
            raise TypeError("Is recording property must be bool or castable to a bool.")

    @property
    def animation_frame_duration(self)->float:
        return self.__animation_frame_duration
    
    @animation_frame_duration.setter
    def animation_frame_duration(self, new_val:float)->None:
        if isinstance(new_val, (float, int)) and new_val > 0:
            self.__animation_frame_duration = new_val
        else:
            raise TypeError("Animation frame duration must be a float or int > 0.")

class LinkedGammaDebugPlotter(tdp):
    def __init__(self):
        #properties specific to this instance
        import smithplot
        from smithplot import SmithAxes

        #update the axes properties
        SmithAxes.update_scParams({"plot.marker.hack": False,
                                "plot.marker.rotate": True,
                                "grid.minor.enable": False,
                                "grid.minor.fancy": False,
                                "axes.normalize.label": False,
                                "grid.major.xmaxn": 9,
                                "axes.xlabel.fancybox": {"boxstyle": "round4,pad=0.25,rounding_size=0.2",
                                                            "facecolor": 'w',
                                                            "edgecolor": "w",
                                                            "mutation_aspect": 0.75,
                                                            "alpha": 0.5},
                                })
        
        #run the parent intialization
        super().__init__()

        self.__def_gamma_meas_plt_args = dict(marker="x", linestyle="none", markersize=10)
        self.__def_gamma_des_plt_args  = dict(marker="o", linestyle="none")

        #plot arguments for the tuners
        self.__gamma_meas_plt_args = []
        self.__gamma_des_plt_args  = []

        #information for tracking current tuning state
        self.__tuner_iterations    = []
        self.__outter_iterations   = None
        self.__last_tuner          = None
        self.__tuner_plt_count     = 0
        self.__vst_plt_count       = 0
        
        #create the axis 
        self.f  = plt.figure()  #create a new figure
        self.ax = plt.subplot(1, 1, 1, projection='smith', axes_xlabel_rotation=90) #make a subplot

    def on_entry(self, src)->None:
        #import the smithplot library
        plt.ion()

    def update(self, src)->None:
        #get the data relevant to the default implementation

        #clear the current axis
        self.ax.cla()

        #we want to plot the call source last so 
        for idx, tuner in enumerate(self.tuners):
            if not tuner is src:
                #update the data for the tuner (calling tuner should be updated)
                tuner.update()
                #now plot the data
                self._plt_src(tuner, at_idx=idx)
        
        #final source should be plotted so it's on top
        self._plt_src(src)

        #update the plot title
        # self.ax.set_title(f"Refining {src.name} Tuner...")
        self.__update_title(src)

        #draw the current plot
        plt.draw()

        #pause so the plot can be updated
        plt.pause(1)
    
    def __update_title(self, pres_src):
        #determine if the current source is the same as the last 
        #plot source

        #get the source index
        src_idx = self.get_tuner_index(pres_src)

        if pres_src is self.__last_tuner:
            #update the tuner plot count
            self.__tuner_plt_count += 1
        else:
            #set the last tuner to the present tuner
            self.__last_tuner = pres_src
            #set the current tuner plot count to 1
            self.__tuner_plt_count = 1
            #if this is the first tuner update the outter loop index
            if src_idx == 0:
                self.__vst_plt_count += 1
            
        #build the title string
        title_str = ""

        #if the vst loops have been given 
        if not (self.__outter_iterations is None):
            title_str = title_str + f"Trial {self.__vst_plt_count} of {self.__outter_iterations}: "
        
        #update the main title string
        title_str = title_str + f"Refining {pres_src.name} Tuner..."
        
        #add the iteration count
        if not (self.__tuner_iterations[src_idx] is None):
            title_str = title_str + f"(Iteration {self.__tuner_plt_count} of {self.__tuner_iterations[src_idx]})"
        
        #now make the title 
        self.ax.set_title(title_str)

    def reset_counts(self):
        #resets the counting states of the plotter
        self.__tuner_plt_count = 0
        self.__vst_plt_count   = 0

    def on_exit(self, src)->None:
        #turn off interactive mode
        # plt.ioff()
        pass

    def register(self, tuner_obj):
        #update the plot arguments 
        self.__gamma_meas_plt_args = self.__gamma_meas_plt_args + [self.__def_gamma_meas_plt_args]
        self.__gamma_des_plt_args  = self.__gamma_des_plt_args + [self.__def_gamma_des_plt_args]

        #update the tuner indices
        self.__tuner_iterations = self.__tuner_iterations + [None]

        #run the super class register method
        return super().register(tuner_obj)
    
    def remove(self, tuner_obj)->int:
        #overload of the remove method. Works essentially the same 
        #but also removes the properties specific to this implementation.

        #remove the tuner and get the index (for other properties)
        idx_for_removal = super().remove(tuner_obj)

        #now remove the properties associated with the tuner
        self.__gamma_meas_plt_args.pop(idx_for_removal)
        self.__gamma_des_plt_args.pop(idx_for_removal)
        self.__tuner_iterations.pop(idx_for_removal)
    
    def savefig(self, fname, *args, **kwargs):
        #Save the figure. See matplotlib.pyplot.savefig for more details
        
        #what is the file type for the save figure
        [_, _, extension] = fileparts(fname)

        #check to see if dpi is being reset
        if ("dpi" in kwargs) and (extension in ('.png', '.jpg')):
            #get the previous dpi setting of the plot 
            pres_dpi = self.f.get_dpi()
            new_dpi  = kwargs["dpi"]

            #get all lines in the figure
            lines = self.ax.lines

            #now for each line update the the figure size
            for idx, line in enumerate(lines):
                #get the previous marker size
                prev_size = line.get_markersize()
                #the new marker size will be 
                new_size = prev_size * (new_dpi / pres_dpi)
                #update the marker size in the plot
                line.set_markersize(new_size)
        
        #make the figure associated with this plotter the current one
        plt.figure(self.f.number) 

        #now have pyplot save the figure
        plt.savefig(fname, *args, **kwargs)

    #supporting methods
    def set_tuner_iterations(self, tuner_ref, new_val):
            #get the tuner index
            tuner_idx = self.get_tuner_index(tuner_ref)
            #make sure the number of iterations is valid
            if isinstance(new_val, (int, float)):
                #typecast to integer
                new_val = int(new_val)
                #now make sure it is > 0
                if new_val > 0:
                    self.__tuner_iterations[tuner_idx] = new_val
                else:
                    raise ValueError("Tuner iterations should be > 0")
            elif new_val is None:
                #this removes the plot
                self.__tuner_iterations[tuner_idx] = new_val
            else:
                raise NotImplementedError("Argument type not implemented for this method.")

    def set_plot_arguments(self, tuner_name, prop_name, **kwargs):
        #Set the plot arguments for the specific tuner. Presently no checks are provided for the 
        #individual plots

        #get the index of the tuner by name
        idx = self.get_tuner_index(tuner_name)
        # idx = self._get_tuner_idx_by_name(tuner_name)

        #now handle the plot argument
        if prop_name == "gamma_meas":
            #set the plot arguments for gamma measured plot
            self.__gamma_meas_plt_args[idx] = kwargs
        elif prop_name == "gamma_des":
            #set the plot arguments for the gamma desired plot
            self.__gamma_des_plt_args[idx] = kwargs
        else:
            raise ValueError(f"Unknown property with name {prop_name}")
        
    def _plt_src(self, src, at_idx:Union[int,None]=None)->None:
        
        #get the index if not provided
        if at_idx is None:
            at_idx = self._get_tuner_idx_by_name(src.name)

        #DEVEL: get the error mask from the tuner
        err_mask = src.sig_mask.receive_mask
        des_gamma = src.target_gamma
        meas_gamma = src.get_tuner_gamma()
        meas_gamma = meas_gamma[err_mask]

        #plot the data
        # self.ax.plot(src.target_gamma, **self.__gamma_des_plt_args[at_idx])
        # self.ax.plot(src.get_tuner_gamma(), **self.__gamma_meas_plt_args[at_idx])
        self.ax.plot(des_gamma, **self.__gamma_des_plt_args[at_idx])
        self.ax.plot(meas_gamma, **self.__gamma_meas_plt_args[at_idx])

    def _get_tuner_idx_by_name(self, tuner_name:str)->int:

        #return the idx of the matching tuner
        for idx, tuner in enumerate(self.tuners):
            if tuner.name == tuner_name:
                return idx
        
        #if we make it to this point, raise an error
        raise RuntimeError(f"Could not find tuner with name {tuner_name}")

    @property
    def tuner_iterations(self)->list[Union[int, None], Union[int, None]]:
        return self.__tuner_iterations

    @property
    def total_iterations(self)->Union[int, None]:
        return self.__outter_iterations
    
    @total_iterations.setter
    def total_iterations(self, new_val:int)->None:
        if isinstance(new_val, (int,float)):
            #make sure it's an integer
            new_val = int(new_val)
            #make sure that the value is greatter than 0
            assert new_val > 0, "Total iterations must be greatter than 0"
            #set the value
            self.__outter_iterations = new_val
        else:
            raise TypeError("Total Iterations must be an integer.")

class LinkedIMDGammaDebugPlotter(LinkedGammaDebugPlotter):
    def __init__(self):
        #run super class initialization
        super().__init__()

        #add in some additional private properties
        self.__def_gamma_meas_plt_args = dict(marker="x", linestyle="none", markersize=10)
        self.__def_gamma_des_plt_args  = dict(marker="o", linestyle="none")

        #plot arguments for the tuners
        self.__gamma_meas_sig_plt_args = []
        self.__gamma_meas_imd_plt_args = []
        self.__gamma_des_plt_args  = []

    def _plt_src(self, src, at_idx:Union[int,None]=None)->None:
        
        #get the index if not provided
        if at_idx is None:
            at_idx = self._get_tuner_idx_by_name(src.name)

        #DEVEL: get the error mask from the tuner
        sig_mask = src.sig_mask
        sig_tones = sig_mask.tone_locations
        imd_tones = np.logical_not(sig_tones)

        #get the error and gamma masks
        err_mask = src.sig_mask.receive_mask
        des_gamma = src.target_gamma
        meas_gamma = src.get_tuner_gamma()

        #TODO: Error mask needs to initialize to all true values. For now, if it is None assume it is all true
        if err_mask is None:
            err_mask = np.full_like(sig_tones, True, np.dtype("bool"))

        #seperate the meseared gamma values into signal and imd tones 
        meas_sig_gamma = meas_gamma[np.logical_and(err_mask, sig_tones)]
        meas_imd_gamma = meas_gamma[np.logical_and(err_mask, imd_tones)]

        #plot the data
        # self.ax.plot(src.target_gamma, **self.__gamma_des_plt_args[at_idx])
        # self.ax.plot(src.get_tuner_gamma(), **self.__gamma_meas_plt_args[at_idx])
        self.ax.plot(des_gamma, **self.__gamma_des_plt_args[at_idx])
        self.ax.plot(meas_sig_gamma, **self.__gamma_meas_sig_plt_args[at_idx])
        if meas_imd_gamma.size > 0:
            #only plot this if it isn't empty.
            self.ax.plot(meas_imd_gamma, **self.__gamma_meas_imd_plt_args[at_idx])

    def register(self, tuner_obj):

        #update the plot arguments 
        self.__gamma_meas_sig_plt_args = self.__gamma_meas_sig_plt_args + [self.__def_gamma_meas_plt_args]
        self.__gamma_meas_imd_plt_args = self.__gamma_meas_imd_plt_args + [self.__def_gamma_meas_plt_args]
        self.__gamma_des_plt_args  = self.__gamma_des_plt_args + [self.__def_gamma_des_plt_args]

        #run super class registration
        return super().register(tuner_obj)
    
    def set_plot_arguments(self, tuner_name, prop_name, **kwargs):
        #Set the plot arguments for the specific tuner. Presently no checks are provided for the 
        #individual plots

        #get the index of the tuner by name
        idx = self.get_tuner_index(tuner_name)
        # idx = self._get_tuner_idx_by_name(tuner_name)

        #now handle the plot argument
        if prop_name == "gamma_meas_sig":
            #set the plot arguments for sig gamma measured plot
            self.__gamma_meas_sig_plt_args[idx] = kwargs
        elif prop_name == "gamma_meas_imd":
            #set the plot arguments for imd gamma measured plot
            self.__gamma_meas_imd_plt_args[idx] = kwargs
        elif prop_name == "gamma_des":
            #set the plot arguments for the gamma desired plot
            self.__gamma_des_plt_args[idx] = kwargs
        else:
            raise ValueError(f"Unknown property with name {prop_name}")

class AnimatedLinkedIMDGammaDebugPlotter(LinkedIMDGammaDebugPlotter, AnimatedDebugPlotter):
    """
    Supports animated plotting of the linked gamma debugger with IMD tone plotting
    NOTE: This is just a developmental class. Should be handled elsewhere

    TODO: Functionality developed here will eventually be merged into the main LinkedGammaDebugPlotter
    class. This is implemented here to keep that class up an running in the meantime
    """

    def __init__(self):

        #run the parent constructors
        LinkedIMDGammaDebugPlotter.__init__(self)
        AnimatedDebugPlotter.__init__(self)

    def update(self, src):
        #overload the update method to bind the two together

        #call the super class update method
        super().update(src)

        #now update the animation 
        self.update_animation(target_figure=self.f)

class AnimatedLinkedGammaDebugPlotter(LinkedGammaDebugPlotter, AnimatedDebugPlotter):
    """
    Supports animated plotting of the linked gamma debugger

    TODO: Functionality developed here will eventually be merged into the main LinkedGammaDebugPlotter
    class. This is implemented here to keep that class up an running in the meantime
    """

    def __init__(self):

        #run the parent constructors
        LinkedGammaDebugPlotter.__init__(self)
        AnimatedDebugPlotter.__init__(self)
    

    def update(self, src):
        #overload the update method to bind the two together

        #call the super class update method
        super().update(src)

        #now update the animation 
        self.update_animation(target_figure=self.f)


#Classes for interfacing with the VST Measurement System
class VST_Tuner_Source():
    def __init__(self, vst, channel_idx, source_obj, excitation_grid:Grid, src_grid:Union[Grid,None]=None):
        #set the receiver's VST
        # if isinstance(vst, VST_Sys):
        #     self.vst = vst
        # else:
        #     raise ValueError("Provided vst is not an instance of the VST system.")
        self.vst = vst
        
        #set the channel index
        if isinstance(channel_idx, int) and (channel_idx in [1, 2]):
            self.channel = channel_idx
        else:
            raise ValueError("Provided index is not valid.")
        
        #set the tuner's source grid
        if isinstance(excitation_grid, Grid):
            self.grid = excitation_grid
        else:
            raise ValueError("Provided index is not a valid Grid")
        
        #set the source grid
        if isinstance(src_grid, Grid):
            self.__src_grid = src_grid
        elif src_grid is None:
            self.__src_grid = None
        else:
            raise TypeError("src_grid must be a Grid or None type.")
        
        #set the source object
        self.__src_obj = source_obj

    def update_tone_set(self, next_tone_set:np.ndarray[bool], ref_grid:Grid)->tuple[int, int]:
        """
        Update the current tones in the VST transmitter. 

        Next tone set should be a boolean array of values the size of the reference grid provided.

        NOTE: Adding and removing tones from the active tone set 
        """
        
        #get the current active tone set
        pres_tone_set = self.active_tones

        #First: determine which tones to add
        #these tones are not in the present set but are in the next set
        tones_to_add = np.logical_and(np.logical_not(pres_tone_set), next_tone_set)

        #if there are any tones to add, add them
        if np.any(tones_to_add):
            #add the tones to the transmitter
            self.add_tones(tones_to_add, on_grid=ref_grid)

        #Next: determine which tones should be removed
        #these tones are in the present set but not in the next set
        tones_to_remove = np.logical_and(pres_tone_set, np.logical_not(next_tone_set))

        #if there are any tones to remove, remove them 
        if np.any(tones_to_remove):
            #remove the tones from the transmitter
            self.rem_tones(tones_to_remove, on_grid=ref_grid)
        
        #return the tones added and removed
        return np.sum(tones_to_add), np.sum(tones_to_remove)

    def add_tones(self, to_tones:np.ndarray[bool], on_grid:Union[Grid, None]=None, 
                  initial_mag:float=-80, initial_phase:float=0.0, phase_is_deg:bool=True):
        """
        Add new tones to the reciever with a specific phase and relative value
        """
        #get the number of tones that will be updated
        num_tones = np.sum(to_tones)

        #if the number of tones to add is zero, don't bother continuing
        if num_tones == 0:
            return
        
        #get the grid to use as reference
        if not isinstance(on_grid, Grid):
            on_grid = self.grid

        #check inputs for initial magnitude
        if isinstance(initial_mag, (int, float)):
            initial_mag = np.full(num_tones, float(initial_mag), dtype=np.dtype("float"))
        elif isinstance(initial_mag, np.ndarray):
            initial_mag = initial_mag.astype("float")
        else:
            raise NotImplementedError("Data type for initial magnitude is not supported.")
        
        #check inputs for initial phase
        if isinstance(initial_phase, (int, float)):
            initial_phase = np.full(num_tones, float(initial_phase), dtype=np.dtype("float"))
        elif isinstance(initial_phase, np.ndarray):
            initial_phase = initial_phase.astype("float")
        else:
            raise NotImplementedError("Data type for initial phase is not supported.")
        
        #push inputs to the vst
        new_vals = polar2complex(db2mag(initial_mag), initial_phase, phase_is_degrees=phase_is_deg)

        #now send this data to the vst bench
        self.vst.add_tones_from_grid(self.__src_obj, to_tones, on_grid, new_vals)
    
    def rem_tones(self, to_tones:np.ndarray[bool], on_grid:Union[Grid, None]=None):
        """
        Remove tones from the present multitone source
        """
        #get the number of tones that will be updated
        num_tones = np.sum(to_tones)

        #if the number of tones to add is zero, don't bother continuing
        if num_tones == 0:
            return
        
        #get the grid to use as reference
        if not isinstance(on_grid, Grid):
            on_grid = self.grid

        #pass these arguments along to the VST bench
        self.vst.rem_tones_from_grid(self.__src_obj, to_tones, on_grid)
        
    def apply_delta(self, newVal, on_grid:Grid=None, to_tones:Union[np.ndarray[bool],None]=None)->None:
        #have the VST driver do most of the work 

        #handle default value of the excitation grid
        if not isinstance(on_grid, Grid):
            on_grid = self.grid #use the entire excitation grid as the default

        #now have the vst apply the source excitation
        self.vst.apply_delta_from_grid(newVal, self.__src_obj, from_grid=on_grid, to_tones=to_tones)
    
    def apply_absolute(self, newVal:np.ndarray[complex], on_grid:Union[Grid,None]=None, keep_vals:Union[np.ndarray[bool],None]=None)->None:
        """
        Apply Absolute update to vst signal from grid
        """

        #handle the default value of the excitation grid
        if not isinstance(on_grid, Grid):
            on_grid = self.grid

        #attempt to update the vst with absolute values
        self.vst.apply_absolute_from_grid(newVal, self.__src_obj, from_grid=on_grid, using_tones=keep_vals)

        #TODO: Iteratively apply updates to the vst to get the correct output power level

    def on(self): 
        self.__src_obj.OutputEnabled=True
    
    def off(self):
        self.__src_obj.OutputEnabled=False

    @property
    def source_object(self):
        #get the source object from the source
        return self.__src_obj 
    
    @property
    def source_grid(self):
        if self.__src_grid is None:
            #assume that the source grid is the parent of the current grid
            return self.grid.parent
        else:
            #return the stored source grid
            return self.__src_grid

    @property
    def active_tones(self):
        #get the active tones directly from the source
        #these will be the tone indices on the source grid (parent grid from the excitation grid)
        tone_idxs = np.array(list(map(int, self.__src_obj.RelativeMultiTones.RelativeFrequencyIndexes)))

        #source grid is the parent grid of the excitation grid (the grid of this object)
        src_grid = self.source_grid

        #now determine the active tones on the src_grid
        tone_idxs = tone_idxs[:,np.newaxis]
        found_tones = tone_idxs == src_grid.index(about_center=True)

        #get the active tones on the src_grid
        active_tones = np.any(found_tones,0)

        #use the mask rather than casting
        return src_grid.cast(active_tones, self.grid, False, dtype=np.dtype("bool"))

class VST_Tuner_Receiver():
    def __init__(self, vst, channel_idx, measurement_grid:Grid,
                  is_tuner_ref = False, alignment_ref=None):
        """
        Initialize the tuner receiver for the VST bench.
        This should be a light-weight abstraction of the VST.
        """
        
        #set the receiver's VST
        # if isinstance(vst, VST_Sys):
        #     self.vst = vst
        # else:
        #     raise ValueError("Provided vst is not an instance of the VST system.")
        self.vst = vst
        
        #set the channel index
        if isinstance(channel_idx, int) and (channel_idx in [1, 2]):
            self.channel = channel_idx
        else:
            #this needs to be better. We need to have a way of 
            raise ValueError("Provided index is not valid.")
        
        #set the measurement grid reference
        if isinstance(measurement_grid, Grid):
            self.grid = measurement_grid
        else:
            raise TypeError("Provided grid must be a valid Grid")
        
        #set designation as tuner reference 
        if is_tuner_ref in [True, False]:
            #set the current receiver as the tuner reference
            self.is_tuner_ref = is_tuner_ref
        else:
            raise ValueError("Provided value for is_tuner_ref must be True or False.")
        
        #set the corresponding tuner as reference
        if isinstance(alignment_ref, ActiveTuner):
            self.alignment_ref = alignment_ref
        else:
            self.alignment_ref = None

        #initialize the aligned values of the tuner
        self.last_grid = self.grid
        self._A = None
        self._B = None

    def measure(self, to_grid:Union[Grid, None]=None):
        # Triggers a measurement of the entire VST system 
        if self.alignment_ref is not None:
            #use the cross-reference technique
            self.vst.measureV3(reference_tuner=self.alignment_ref, perform_alignment=True, use_expected_source=False)
        else:
            #use the original approach
            self.vst.measureV2(perform_alignment=True, use_expected_source=self.is_tuner_ref)

        #if not set the requested grid will be the measurement grid
        self.import_to_grid(to_grid)
        
    def import_to_grid(self, grid:Union[Grid, None]=None):
        #if not set the requested grid will be the measurement grid
        if not isinstance(grid, Grid):
            self._A = self.A
            self._B = self.B
            self.last_grid = self.grid
        else:            
            #now update the aligned values of the tuner
            self._A =  self.grid.cast(self.a, grid)
            self._B =  self.grid.cast(self.b, grid)
            self.last_grid = grid
        
    @property 
    def measurement_idx(self):
        return (self.channel - 1) * 2

    @property 
    def a(self):
        #return the unaligned, measured a-wave
        return self.vst.measuredSpectra[self.measurement_idx]
    
    @property 
    def b(self):
        #return the unaligned, measured b-wave
        return self.vst.measuredSpectra[self.measurement_idx + 1]
    
    @property 
    def A(self):
        #return the aligned, measured a-wave
        return self._A
    
    @property 
    def B(self):
        #return the aligned, measured b-wave
        return self._B

class VST_Active_Tuner(ActiveTuner):       
    def __init__(self, 
                 name:str, 
                 rf_source: VST_Tuner_Source, 
                 rf_receiver: VST_Tuner_Receiver, 
                 tuner_excitation_grid: Grid, 
                 extern_plotter:Union[tdp,None]=None,
                 extern_masker=None):
        """
        To initialize the active tuner the following three objects are required:
            1) An interface to a RF source
            2) An interface to a RF receiver
            3) A definition of the excitation grid
        """

        #call the superclass initialization
        super().__init__(debug_plotter=extern_plotter, signal_masker=extern_masker)

        #a name is required for the plotting interface
        if isinstance(name, str):
            self.name = name

        #now set the source, receiver, and grid definition objects
        if isinstance(rf_source, VST_Tuner_Source):
            self.Source = rf_source
        else:
            raise ValueError("rf_source must be an instance of the VST_Source class.")
        

        #for the receiver
        if isinstance(rf_receiver, VST_Tuner_Receiver):
            self.Receiver = rf_receiver
        else:
            raise ValueError("rf_reciever must be an instance of the VST_Reciever class.")
        
        #defines the tuner excitation grid
        if isinstance(tuner_excitation_grid, Grid):
            self.grid = tuner_excitation_grid
        else:
            raise ValueError("tuner_excitation_grid must be an instance of Grid")

        #now define the hidden properties used by this subclass
        self.__gamma_0       = None
        self.__gamma_in_0    = None
        self.__A0            = None
        self.__A0_expected   = None #to be removed 
        self.__B0            = None
        self.__S0            = None
        self.__T0            = None

        #Developemental parameters
        self.__noise_floor      = -70   #noise floor in dBm
        self.__excitation_mask  = None  #mask of tone values (np.ndarray[bool])
        # self.extern_excitation_mask = None
        self.__error_mask       = None  #mask of error values


        # #additional arguments for plot bindings (for sharing plots for multiple tuners)
        # self.__plotter                     = None
        # This is handled by the tuner subclass
    
    # Private methods for this class
    def __is_valid_tuner_grid_prop(self, newProperty:np.ndarray) -> bool:
        return self.grid.is_compatible(newProperty)

    def __is_valid_tuner_signal(self, newSignal:MultitoneSignal) -> bool:
        #check to see if the signal shares the same root with the tuner
        return self.grid.is_compatible(newSignal.grid)

    # Abstract ActiveTuner method definitions
    def _is_target_gamma_valid(self, new_gamma):
        #implementation check for the target gamma
        return (self.grid.is_compatible(new_gamma) and 
                (new_gamma.dtype is np.dtype("complex")))

    def measure(self):
        """
        This method depends on the exact implementation of this class.
        This method should trigger a new A/B wave measurement. It is assumed
        that the the resulting A and B waves are pre-aligned when being brought 
        into the tuner. 
        """

        #raise a warning if the current tuner hasn't been initialized.
        if not self.is_initialized:
            warnings.warn("Tuner has not been initialized.")

        #have the reciever perform a measurement
        self.Receiver.measure(to_grid=self.grid)

    def generate_source_update(self):
        """
        This generates an update vector for the transmitter. The exact way this
        is accomplished depends on the implementation. The suggested approach is:

        return self.A0_expected / self.A0
        """
        #raise a warning if the current tuner hasn't been initialized.
        if not self.is_initialized:
            raise RuntimeError("Tuner has not been initialized. Please initialize before generating a source update.")

        return self.A0_expected / self.A0
    
    def initialize(self, A0:Union[np.ndarray,None]=None, S0:Union[np.ndarray,None]=None,
                    gamma_0:Union[np.ndarray,None]=None, gamma_in_0:Union[np.ndarray,None]=None,
                    target_gamma:Union[np.ndarray, None]=None)-> None: 
        #Run the initialization
        
        #for gamma_0
        if not (gamma_0 is None):
            self.gamma_0 = gamma_0
        elif not isinstance(self.gamma_0, np.ndarray):
            raise RuntimeError("gamma_0 hasn't been provided or set. Cannot initialize.")
        
        #for gamma_in_0
        if not (gamma_in_0 is None):
            self.gamma_in_0 = gamma_in_0
        elif not isinstance(self.gamma_in_0, np.ndarray):
            raise RuntimeError("gamma_in_0 hasn't been provided or set. Cannot initialize.")
        
        #for A0 
        if not (A0 is None): 
            self.A0 = A0
        elif not isinstance(self.A0, np.ndarray):
            raise RuntimeError("A0 hasn't been provided or set. Cannot initialize.")

        #for S0 
        if isinstance(S0,(np.ndarray,MultitoneSignal)) :
            self.S0 = S0
        elif isinstance(self.S0, np.ndarray): 
            raise TypeError("S0 hasn't been provided or set. Cannot initialize.")
        
        #for initial target 
        if not (target_gamma is None):
            #use the initial target provided by the user
            self.target_gamma = target_gamma
        elif self.target_gamma is None:
            #if the initial target has been set, initialize to the center of the smith chart
            self.target_gamma = np.full(self.grid.size, 0, dtype=np.dtype("complex"))

        #generate initial values for T0 and A0_expected
        self.T0 = self.A0 - self.S0
        # self.A0_expected = self.A0.copy()
        
        #mark as initialized 
        self.is_initialized = True

    def shutdown(self):
        #Shutdown the tuner

        #directly set the hidden values of the signals and gammas 
        self.__A0            = None
        self.__A0_expected   = None
        self.__B0            = None
        self.__S0            = None
        self.__T0            = None

        #set the initialization variable to false
        self.is_initialized = False

    def get_active_tones(self, on_grid:Union[Grid, str, None])->np.ndarray[bool]:
        """ 
        get_active_tones:
            Return the active tone set for this tuner on the requested grid
            if no grid is specified the active tones are returned on the tuner 
            grid
        """

        #get the active tones directly from the source
        active_tones = self.Source.active_tones

        #now cast to the requrested grid
        if on_grid is None:
            on_grid = self.grid
        elif isinstance(on_grid,str):
            on_grid = self.get_grid_by_name(on_grid, raise_error_if_not_found=True)
        elif not isinstance(on_grid, Grid):
            raise TypeError("Provided grid must be a Grid, name of a grid, or None")

        #return the tones after casting to the requested grid
        return self.grid.cast(active_tones, on_grid, off_grid_vals=False, dtype="bool")
        
    def get_abwaves(self, on_grid:Union[Grid, None])->tuple[np.ndarray, np.ndarray]:

        #now set the a and b waves from the tuner's receiver
        #note that these may not be properly aligned anymore 
        #but this shouldn't be a problem here.
        A = self.Receiver.a
        B = self.Receiver.b

        #the receiver's grid is always the grid to cast from
        rx_grid = self.Receiver.grid

        #cast the data to a new grid, if provided
        if isinstance(on_grid, Grid):
            A = rx_grid.cast(A, on_grid)
            B = rx_grid.cast(B, on_grid)

        elif isinstance(on_grid, str):

            #attempt to get the grid by name
            on_grid = self.get_grid_by_name(on_grid, raise_error_if_not_found=True)

            #cast to the requested grid
            A = rx_grid.cast(A, on_grid)
            B = rx_grid.cast(B, on_grid)
        
        #also return the frequencies of the grid
        freqs = on_grid.freqs
        
        #return the waves
        return A, B, freqs

    def get_grid_by_name(self, grid_name:str, raise_error_if_not_found:bool=False)->Grid:
        #finds the grid specific to the tuner implementaiton. 
        #Returns NoneType if not found

        #makes sure that the grid name is a string
        if not isinstance(grid_name, str):
            raise TypeError("Grid name must be a string.")
        else:
            grid_name = grid_name.lower()
        
        #remove capitolization from grid name
        grid_name = grid_name.lower()

        #if it is, then find the grid name
        if grid_name in ("rx", "receiver", "measurement"):
            #return the 
            return self.Receiver.grid
        elif grid_name == "tuner":
            #return the grid for the tuner
            return self.grid
        elif grid_name in ("source", "tx", "transmitter", "excitation"):
            #return the grid for the transmitter, same as 
            return self.Source.grid
        elif (grid_name == "signal") and isinstance(self.__signal, MultitoneSignal):
            #return the grid for the signal. The signal may be a multitone signal object
            #or an array. If it's an array, throw an error.
            return self.__signal.grid
        else:
            if raise_error_if_not_found:
                raise ValueError(f"Could not find grid with name {grid_name}")
            else:
                return None
            
    ## Developemental methods
    def gen_A0_mask(self):
        """
        Generate the Mask for A0 Values
        """

        #get the tone power 
        P0 = np.abs(self.A0_expected)**2 / (2 * 50)

        #convert to dBm 
        P0 = 10*np.log10(P0) + 30

        #return a mask of values above the threshold
        return P0 > self.__noise_floor 
    
    def gen_B_mask(self, include_excitation_mask:bool=True)->np.ndarray[bool]:
        """
        Generate a mask for B values of relevance
        """
        #get the b-tones
        b_tones = self.B
        
        #get the power of each tone
        P0 = np.abs(b_tones)**2 / (2 * 50)
        
        #convert to dBm
        P0 = 10*np.log10(P0) + 30

        #return a mask of values that were above the noise floor
        b_mask = P0 > self.__noise_floor

        #optionally and it with the A0_mask
        if include_excitation_mask:
            b_mask = np.logical_and(b_mask, self.__excitation_mask)

        #return the final mask
        return b_mask

    def apply_present_excitation(self)->None:
        """
        Update the source with the present excitation from the tuner

        Absolute updates to the source have been removed in favor of relative only source updates. 
        """

        #get the new excitation and transmit mask
        self.__excitation_mask  = self.sig_mask.transmit_mask

        #apply the transmit mask to the source
        tones_added, tones_removed = self.Source.update_tone_set(self.__excitation_mask, self.grid)

        #has the signal changed?
        signal_changed = (tones_added + tones_removed) > 0

        #if tones were added or removed, remeasure and generate a new update
        if signal_changed:
            #renew the states of the signal for the relative update
            self.update_signal_states()
            #update source excitation
            # self.self.update_T0()

        #get the value of the delta vector to apply
        source_update = self.generate_source_update()

        #apply the change
        self.apply(source_update, type="relative")

    def apply(self, newVal:np.ndarray[complex], type:str="relative")->None:
        """
        Sets the new tuner excitation.
        """
        #raise a warning if the current tuner hasn't been initialized.
        if not self.is_initialized:
            warnings.warn("Tuner has not been initialized.")

        #make sure that the type of source update is the correct type
        if not type in ["relative", "absolute"]:
            raise ValueError("type parameter must be relative or absolute")
        
        # if self.extern_excitation_mask is None:
        #     self.extern_excitation_mask = np.full(self.__excitation_mask.size, True)

        if type == "relative":
            #apply a relative change in the tone values
            # self.Source.apply_delta(newVal, self.grid, to_tones=np.logical_and(self.__excitation_mask, self.extern_excitation_mask))
            self.Source.apply_delta(newVal, self.grid, to_tones=self.__excitation_mask)
        else:
            #apply an absolute update to the signal
            # self.Source.apply_absolute(newVal, self.grid, keep_vals=np.logical_and(self.__excitation_mask, self.extern_excitation_mask))
            self.Source.apply_absolute(newVal, self.grid, keep_vals=self.__excitation_mask)
    
    # Other methods specific to this implementation
    def update(self)->None:
        #This triggers an update of the a/b waves stored in the tuner from the source
        #without re-measuring. Note that this does not garuntee that the a/b waves will
        #be aligned. 
        self.Receiver.import_to_grid(self.grid)

    def save(self,
             file_path:Union[str,None]=None)->None:
        #Save the present tuner state variables to the specified file path.
        #Step 1: If the file path is not provided have the user provide it through the UI
        if file_path is None:
                file_path = uiputfile(f"Select destination for tuner data.", filetypes=[("Numpy File", "*.npz")])

        #Step 2: Save the tuner data
        #active tone markers
        freqs = self.Receiver.grid.freqs
        tones = self.grid.cast(self.grid.full_like(True,dtype="bool"), self.Receiver.grid)

        #save the data
        np.savez(file_path,
                 a=self.Receiver.a, b=self.Receiver.b,
                 A0=self.A0, T0=self.T0, S0=self.S0, B0=self.B0,
                 target_gamma=self.target_gamma, measured_gamma=self.get_tuner_gamma(),
                 gamma_in_0=self.gamma_in_0, gamma_0=self.gamma_0,
                 freqs=freqs, tones=tones)
        
    # Abstract ActiveTuner property definitions
    @property
    def A(self)->np.ndarray:
        if self.Receiver.last_grid is self.grid:
            return self.Receiver.A
        else:
            raise TypeError("Cannot pull in the requested A-wave as the receiver ran on another grid. Please re-measure or run import_to_grid.")

    @property 
    def B(self)->np.ndarray:
        if self.Receiver.last_grid is self.grid:
            return self.Receiver.B
        else:
            raise TypeError("Cannot pull in the requested B-wave as the receiver ran on another grid. Please re-measure or run import_to_grid.")

    @property 
    def gamma_0(self)->np.ndarray:
        return self.__gamma_0

    @gamma_0.setter
    def gamma_0(self, newVal)->None:
        # Set the tuner's reflection coefficient

        #if it fits on the grid, set it
        if self.__is_valid_tuner_grid_prop(newVal):
            #set the value
            self.__gamma_0 = newVal
        else:
            raise ValueError("Unable to set gamma_0. Please make sure it is valid on the current tuner grid.")

    @property 
    def gamma_in_0(self)->np.ndarray:
        return self.__gamma_in_0
    
    @gamma_in_0.setter
    def gamma_in_0(self, newVal)->None:
        # Set the dut's reflection coefficient

        #if it fits on the grid, set it
        if self.__is_valid_tuner_grid_prop(newVal):
            #set the value
            self.__gamma_in_0 = newVal
        else:
            raise ValueError("Unable to set the new value of gamma_in_0. Please make sure it is valid on the current tuner grid.")
    
    @property
    def A0(self)->np.ndarray:
        return self.__A0
    
    @A0.setter
    def A0(self, newVal)->None:
        # Set the new value of A0

        #if it fits on the grid, set it
        if self.__is_valid_tuner_grid_prop(newVal):
            #set the value
            self.__A0 = newVal
        else:
            raise ValueError("Unable to set the new value of A0. Please make sure it is valid on the current tuner grid.")
    
    @property
    def B0(self)->np.ndarray:
        return self.__B0
    
    @B0.setter
    def B0(self, newVal)->None:
        # Set the new value of B0

        #if it fits on the grid, set it
        if self.__is_valid_tuner_grid_prop(newVal):
            #set the value
            self.__B0 = newVal
        else:
            raise ValueError("Unable to set the new value of B0. Please make sure it is valid on the current tuner grid.")  

    @property
    def signal_obj(self)->MultitoneSignal:
        # Return the signal object for the current tuner
        return self.__signal

    @property
    def S0(self)->np.ndarray:
        #return the signal excitation
        if self.__signal is None:
            s0 = self.__S0
        else:
            #update the values directly from the stored signal. This is the desired method 
            s0 = self.__signal.grid.cast(self.__signal.v0, self.grid, off_grid_vals=0, dtype=np.dtype("complex"))

        #now return s0
        return s0
    
    @S0.setter
    def S0(self, newVal)->None:
        # Set the new value of B0

        #if it fits on the grid, set it
        if isinstance(newVal,MultitoneSignal) and self.__is_valid_tuner_signal(newVal):
            self.__S0 = None
            self.__signal = newVal
        elif isinstance(newVal, np.ndarray) and self.__is_valid_tuner_grid_prop(newVal):
            #set the value
            self.__S0 = newVal
            self.__signal = None
        else:
            raise ValueError("Unable to set the new value of S0. Please make sure it is valid on the current tuner grid.")

    @property
    def T0(self)->np.ndarray:
        return self.__T0
    
    @T0.setter
    def T0(self, newVal)->None:
        # Set the new value of T0

        #if it fits on the grid, set it
        if self.__is_valid_tuner_grid_prop(newVal):
            #set the value
            self.__T0 = newVal
        else:
            raise ValueError("Unable to set the new value of T0. Please make sure it is valid on the current tuner grid.")
    
    @property
    def gamma_error(self)->float:
        #returns the error of the current tuner
        gamma_meas = self.get_tuner_gamma()
        gamma_des = self.target_gamma

        #update the receive mask from the available data
        error_mask = self.sig_mask.update_receive_mask()

        #return the rms error of the two gamma values
        return np.sqrt(np.mean(np.abs(gamma_meas[error_mask] - gamma_des[error_mask])**2))
    
    @property 
    def a0_error(self)->float:
        #returns the error of the current a0 excitation
        a0_meas = self.A0
        a0_des  = self.A0_expected

        #return the rms error of the excitation
        return np.sqrt(np.mean(np.abs(a0_meas - a0_des)**2))

    @property
    def excitation_mask(self)->np.ndarray[bool]:
        if self.__excitation_mask is None:
            return np.full(self.grid.size, True)
        else:
            return self.__excitation_mask
    
    @property
    def error_mask(self)->np.ndarray[bool]:
        if self.__error_mask is None:
            return np.full(self.grid.size, True)
        else:
            return self.__error_mask
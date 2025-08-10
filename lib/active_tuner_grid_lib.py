import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
import warnings
from typing import Union
from functools import singledispatchmethod

#Grid type enumerations
class GridType(Enum):
    STAND_ALONE  = 0
    ROOT         = 1
    CHILD        = 2

#work-around for decorators
class _Grid:
    """
    Purpose of this class is allow decorators to type check inputs to a grid function
    for other grids. Otherwise it won't work. 
    """
    def __init__():
        pass

#abstract grid source definition
class GridSource:
    def __init__(self):
        pass
    
    @abstractmethod
    def build_hook(self):
        #This is where the mask, reference_frequency, and offsets should be built
        pass
    
    def build(self):
        #this method requires definition in other grid sources
        #this should return the following properties
        #   1) a grid mask,
        #   2) grid reference frequency, and  
        #   3) grid offset frequencies
        
        #run the subclassbuildhook
        mask, ref, offsets = self.build_hook()

        #return the final values
        return mask, ref, offsets
    
class LinspaceSource(GridSource):
    
    @staticmethod
    def __is_valid_reference_type(ref_type):
        if isinstance(ref_type, float):
            #interpret this as a frequency
            return ref_type >= 0 
        else:
            return False
        
    @staticmethod
    def __is_valid_frequency_type(freq_type):
        return freq_type in ["relative", "absolute"]

    def __init__(self, start: float, stop: float, npts: int, frequency_type:str = "absolute", reference=0):
        
        #make sure the frequency type is valid
        if LinspaceSource.__is_valid_frequency_type(frequency_type):
            self.freq_type = frequency_type
        else:
            raise ValueError("frequency_type must be absolute or relative")
        
        #set the reference type
        if LinspaceSource.__is_valid_reference_type(reference):
            self.reference = reference
        else:
            raise ValueError("reference must be start, center, stop, or a floating point number representing the frequency")
        
        #set the start and stop frequencies
        if start < stop:
            self.start = start
            self.stop  = stop
        else:
            raise ValueError("start is not less than stop")
        
        #set the number of points
        if npts > 2:
            self.npts = npts
        else:
            raise ValueError("npts must be > 2")

    def __get_frequencies_as_absolute(self):
        if self.freq_type == "relative":
            #the start and stop values are relative and the reference
            absolute_start = self.start * self.reference
            absolute_stop = self.stop * self.reference
        else: #they are already absolute values
            absolute_start = self.start
            absolute_stop = self.stop
        
        return absolute_start, absolute_stop
    
    def build_hook(self):
        #build the frequency grid as absolute values
        start, stop = self.__get_frequencies_as_absolute()
        freqs = np.linspace(start, stop, self.npts)

        #warn the user if the center frequency won't be on grid
        if self.npts % 2 == 0: #if the number of points is even
            warnings.warn("The center frequency won't be on grid.") 
        
        #build the initial mask
        mask = np.full_like(freqs, True)

        #now build the offsets
        offsets = freqs - self.reference

        #finally, return
        return mask, self.reference, offsets

class FrequencyGridSource(GridSource):
    @staticmethod
    def __is_valid_center_freq(fc):
        return fc > 0
    
    @staticmethod
    def __is_valid_size(size):
        return size > 2
    
    @staticmethod
    def __is_valid_step(step):
        return step > 0

    def __init__(self, about:float = 1e9, size:int = 300, step:float = 5e6):
        
        if FrequencyGridSource.__is_valid_center_freq(about):
            self.ref = about
        else:
            raise ValueError("Center frequency is not valid")
        
        if FrequencyGridSource.__is_valid_size(size):
            self.size = size
        else:
            raise ValueError("Grid size is not valid")
        
        if FrequencyGridSource.__is_valid_step(step):
            self.step = step
        else:
            raise ValueError("Step size is not valid")
        
    def build_hook(self):
        #get the starting and stopping frequencies of the grid
        if self.size % 2 == 0: #even number of steps, i.e. center is off grid
            f_stop = self.ref + (self.size / 2) * self.step  -  (self.step / 2)
            f_start = self.ref - (self.size / 2) * self.step  +  (self.step / 2)
        else: #odd number of steps, i.e. center is on grid
            f_stop = self.ref + ((self.size - 1) / 2) * self.step
            f_start = self.ref - ((self.size - 1) / 2) * self.step

        #now build the frequency grid
        freqs = np.linspace(f_start, f_stop, self.size)

        #now build the mask and offsets 
        mask = np.full_like(freqs, True)
        offsets = freqs - self.ref

        #return
        return mask, self.ref, offsets

class StaticGridSource(GridSource):
    @staticmethod
    def __is_valid_ref_freq(fc):
        return fc > 0

    def __init__(self, freqs: np.ndarray, center=None):

        #if a center frequency is not provided, interpret the frequency
        #array as absolute values
        self.freqs_are_absolute = (center is None)
        
        if self.freqs_are_absolute:
            #the center frequency will be interpreted as the mean frequency
            self.ref = np.mean(freqs)
        elif FrequencyGridSource.__is_valid_ref_freq(center):
            self.ref = center
        else:
            raise ValueError("center argument is not valid.")
        
        #set the frequencies
        self.freqs = freqs

    def build_hook(self):
        #get the starting and stopping frequencies of the grid
        if self.freqs_are_absolute:
            offsets = self.freqs - self.ref
        else: #frequencies are relative to the reference
            offsets = self.freqs

        #now build the mask
        mask = np.full_like(offsets, True)

        #return
        return mask, self.ref, offsets

class GridMod:
    def __init__(self):
        pass

    @abstractmethod
    def mod_hook(self, mask: np.ndarray, reference_freq: float, offset_frequencies: np.ndarray, grid_type: GridType):
        #this is where the subclass will perform the modification
        pass

    def modify(self, mask: np.ndarray, reference_frequency: float, offset_frequencies: np.ndarray, grid_type: GridType):
        #this is a required method for modifying grid.
        #It will take the following four input arguments
        #   mask - the present mask of the grid
        #   reference_frequency - the present reference frequency of the grid
        #   offset_frequencies - the present offset frequencies of the grid
        #   grid_type - the type of grid being generated
        #This method will return the following arguments
        #   mask - Modified mask 
        #   reference_frequency - the reference frequency  
        #   offset_frequencies -  the offset frequencies

        #run the subclass hook 
        mask, reference_frequency, offset_frequencies = self.mod_hook(mask, reference_frequency, offset_frequencies, grid_type)

        #return the values
        return mask, reference_frequency, offset_frequencies

class ApplyBandMask(GridMod):
    """
    Frequency Mask Grid Modifier:
    This is a simple modifier that applies a mask between starting and stopping frequencies. If the new mask is a root mask,
    a mask will also be applied to the offset frequencies.
    """
    @staticmethod
    def __is_valid_freq_type(freq_type: str):
        return freq_type in ["relative", "absolute"]
    
    @staticmethod
    def __is_valid_mask_type(mask_type: str):
        return mask_type in ["keep_in", "keep_out"]
    
    def __init__(self, start: float, stop: float, freq_type="absolute", mask_type="keep_in"):
        
        #check if the frequency type is valid
        if ApplyBandMask.__is_valid_freq_type(freq_type):
            self.freq_type = freq_type
        else:
            raise ValueError("Frequency type argument must be relative or absolute")
        
        #check if the mask type argument is valid
        if ApplyBandMask.__is_valid_mask_type(mask_type):
            self.mask_type = mask_type
        else:
            raise ValueError("Mask type argument should be keep_in or keep_out")

        #now set the start and stop frequencies
        if start < stop:
            self.start = start
            self.stop = stop
        else:
            raise ValueError("Start frequency must be less than stop frequency")

    def __conv_to_freq_type(self, ref_freq: float, offsets: np.ndarray):
        if self.freq_type == "relative":
            return offsets
        else:
            return ref_freq + offsets

    def mod_hook(self, mask: np.ndarray, reference_freq: float, offset_frequencies: np.ndarray, grid_type: GridType):
        
        #get the current frequencies to and the current mask with
        freqs = self.__conv_to_freq_type(reference_freq, offset_frequencies)
        
        #generate the band mask
        band_mask = np.logical_and((self.start <= freqs), (freqs <= self.stop))

        #the mask and the frequency array should be the same size
        if self.mask_type == "keep_out":
            band_mask = np.logical_not(band_mask)

        #generate the new mask 
        mask_out = np.logical_and(mask, band_mask)

        #now if the type is root apply the mask to the offset frequencies
        if grid_type == GridType.ROOT:
            offset_frequencies = offset_frequencies[mask_out]

        #this method should not modify the offset frequencies or reference frequency in any way
        return mask_out, reference_freq, offset_frequencies

class ApplyBooleanOperation(GridMod):
    """
    This is more for applying an arbitrary sub-mask on-top of a known grid. It will automatically fail if you try to apply it to a 
    """
    @staticmethod
    def __is_valid_logical_function(func_str):
        return func_str in ["and","or","not","xor","nand"]

    @staticmethod
    def __is_valid_mask(mask):
        return isinstance(mask,np.ndarray) and isinstance(mask.dtype, np.dtype("bool"))

    def __init__(self, mask:np.ndarray, operation:str="and"):
        
        if ApplyBooleanOperation.__is_valid_mask(mask):
            self.mask = mask
        else:
            raise TypeError("Mask type provided must be a np.ndarray of type bool")
        
        if ApplyBooleanOperation.__is_valid_logical_function(operation):
            self.operation = operation
        else:
            raise TypeError("Operation must be specified as string: and, or, not, xor, nand")

    def __apply_operation(self,mask_in):
        if self.operation == "and":
            #apply and operation
            return np.logical_and(self.mask, mask_in)
        elif self.operation == "or":
            #apply or operation
            return np.logical_or(self.mask, mask_in)
        elif self.operation == "not":
            #apply not operation to input mask
            return np.logical_not(mask_in)
        elif self.operation == "xor":
            #apply xor operation
            return np.logical_xor(self.mask, mask_in)
        elif self.operation == "nand":
            #apply nand operation
            return np.logical_not(np.logical_and(self.mask, mask_in))
        else:
            raise RuntimeError(f"Logical operation keyword {self.operation} not recognized")

    def mod_hook(self, mask: np.ndarray, reference_freq: float, offset_frequencies: np.ndarray, grid_type: GridType):
        
        #fail if the grid type is Root
        if grid_type == GridType.ROOT:
            raise TypeError("ApplyArbMask modifier cannot run on a root grid.")
        
        #make sure that the arbitrary mask size is the same size as the grid mask
        if not self.mask.size == mask.size:
            raise TypeError(f"Mask provided has size {self.mask.size} and the target mask has size {mask.size}")
        
        #update the current mask by logically anding 
        mask = self.__apply_operation(mask)

        #return all the values
        return mask, reference_freq, offset_frequencies

class ExistsAtFrequencies(GridMod):
    def __init__(self, freqs:np.ndarray, interpret_as_absolute:bool=True, error_on_mismatch:bool=False, warn_on_mismatch:bool=False):
        self.target_freqs = freqs 
        self.is_absolute = interpret_as_absolute
        self.error_on_mismatch = error_on_mismatch
        self.warn_on_mismatch = warn_on_mismatch

    def mod_hook(self, mask, reference_freq, offset_frequencies, grid_type):
        #The mask will be applied to the parent frequencies

        #convert to absolute frequencies
        if not self.is_absolute: 
            target_freqs = self.target_freqs + reference_freq
        else:
            target_freqs = self.target_freqs

        #generate the input frequencies
        freqs_in = reference_freq + offset_frequencies

        #build the mask out
        mask_out = np.full_like(mask, False)

        #Get the locations of the target frequencies in 
        exists_at = np.where(np.isin(freqs_in, target_freqs))[0]

        #if exists_at size does not equal that of target frequencies,
        #some of the frequencies didn't mask correctly
        if not (exists_at.size == target_freqs.size):
            if self.error_on_mismatch:
                raise RuntimeError("Could not fully match target mask frequencies with base grid frequencies.")
            elif self.warn_on_mismatch: #throw warning instead and continue
                warnings.warn("Could not fully match target mask frequencies with base grid frequencies.")

        #update the mask
        mask_out[exists_at] = True

        return mask_out, reference_freq, offset_frequencies

class OrMask(GridMod):
    """
    This is more for applying an arbitrary sub-mask on-top of a known grid. It will automatically fail if you try to apply it to a 
    """
    def __init__(self, mask:np.ndarray):
        #add a new mask to apply
        self.mask = mask


    def mod_hook(self, mask: np.ndarray, reference_freq: float, offset_frequencies: np.ndarray, grid_type: GridType):
        
        #fail if the grid type is Root
        if grid_type == GridType.ROOT:
            raise TypeError("ApplyArbMask modifier cannot run on a root grid.")
        
        #make sure that the arbitrary mask size is the same size as the grid mask
        if not self.mask.size == mask.size:
            raise TypeError(f"Mask provided has size {self.mask.size} and the target mask has size {mask.size}")
        
        #update the current mask by logically anding 
        mask = np.logical_and(mask, self.mask)

        #return all the values
        return mask, reference_freq, offset_frequencies

#grid generator definitions
class GridGenerator:
    @staticmethod
    def __is_valid_source_type(arg):
        #return true if the argument is a valid grid source
        return isinstance(arg, GridSource) or (arg is None)
    
    @staticmethod
    def __is_valid_modifier_type(arg):
        #return true if the argument is a valid grid modifier
        return isinstance(arg, GridMod)

    def __init__(self, grid_source:Union[GridSource, None], *argv):
        #Initialize the grid generator

        #initialize the grid modifiers list
        self.modifiers = []
        self.source    = None
        
        #first argument should be the grid source
        if (grid_source is None) or (isinstance(grid_source, GridSource)):
            self.source = grid_source
        else:
            raise TypeError("grid_source argument must be None or a valid GridSource")

        #follwing arguments should be grid modifiers
        for arg in argv:
            #make sure that the argument is a valid GridModifier
            self.append_modifier(arg)
    
    def set_source(self, new_source):
        #make sure that the new source is valid
        if GridGenerator.__is_valid_source_type(new_source):
            #get the previous source
            prev_source = self.source
            #set the new source
            self.source = new_source
            #return the previous source 
            return prev_source
        else:
            raise TypeError("")

    def append_modifier(self, new_mod):
        #append a new modifier (if valid)
        if GridGenerator.__is_valid_modifier_type(new_mod):
            self.modifiers.append(new_mod)
        else:
            raise TypeError("Provided argument is not a valid modifier")
    
    #property definitions for grid generator
    @property
    def num_modifiers(self):
        #returns the generator's number of modifiers
        return len(self.modifiers)

    def generate(self, name: str, type: GridType, source_grid=None):
        #run the grid generation routine 

        #if a source grid has been provided, it will act as the generator
        if source_grid is None:
            if not self.source is None : #source grid must be type GridSource
                mask, ref, offsets = self.source.build()
            else:#if it's still None throw and error
                raise RuntimeError("Grid source is not set. Cannot generate a new grid")
        elif isinstance(source_grid, Grid):
            #get the three grid properties from the source
            mask = np.full(np.sum(source_grid.mask), True) #should only be the true values in parent grid's mask
            ref = source_grid.ref_freq
            offsets = source_grid.offset_freqs
        else:
            raise TypeError("Provided grid is not valid")
        
        #apply the grid modifiers
        for mod in self.modifiers:
            mask, ref, offsets = mod.modify(mask, ref, offsets, type)
        
        #return the new grid
        return Grid(name, type, parent=source_grid, mask=mask, ref=ref, offsets=offsets)

class Grid(_Grid):
    @staticmethod
    def generate(name: str, using:Union[None,GridGenerator]=None, on=None):
        """
        Method for generating a grid. 
        This method only generates a child or a root frequency grid. 
        """
        
        #the using argument must be a type of grid generator
        if (not using is None) and (not isinstance(using, GridGenerator)):
            raise TypeError("The provided grid generator must be None or of type GridGenerator.")

        #make sure that on is a type of grid
        if (not on is None) and (not isinstance(on, Grid)):
            raise TypeError("Cannot generate grid on object provided as it isn't a grid itself.")

        #determine the type of the new frequency grid
        if on is None:
            new_type = GridType.ROOT
        else:
            new_type = GridType.CHILD

        #attempt to generate and return the new grid
        if not (using is None):
            return using.generate(name, new_type, on)
        else:
            raise NotImplementedError("Default generator not implemented yet.")

    @staticmethod
    def __is_valid_frequency_array(arrayIn: np.ndarray):
        return (arrayIn.ndim == 1) and (arrayIn.dtype is np.dtype("float64"))
    
    @staticmethod
    def __is_valid_mask(arrayIn: np.ndarray):
        #determines whether the mask is a valid mask type
        return (arrayIn.ndim == 1) and (arrayIn.dtype is np.dtype("bool"))
    
    @staticmethod
    def __is_valid_parent(new_parent):
        return  (isinstance(new_parent, Grid) and                       #must be an instance of a grid
                 (not new_parent.type == GridType.STAND_ALONE) and      #must not be stand alone grid
                 (new_parent.__has_root()))                             #must have a valid root node
    
    def __has_root(self):
        if self.has_parent: #if there is a parent, check the parent out                                              
            return self.__parent.__has_root()                   
        elif self.type == GridType.ROOT: #assumes that the node doesn't have a parent
            return True
        else: #otherwise there isn't a proper root grid
            return False
    
    def __shares_root_with(self, other_grid):
        #checks if two grids have the same root grid.
        #it's assumed that we have already checked that the other_grid is of type Grid
        return self.root is other_grid.root
    
    def __is_valid_child(self, new_child):
        return ((new_child.type == GridType.CHILD) and          #Needs to be type child
                (np.sum(self.mask) == new_child.mask.size))     #Child mask needs to be the same size as all true values in the parent

    def __init__(self, name: str, type: GridType, parent=None, mask=None, ref=None, offsets=None):
        """
        Initialize a new grid
        """
        #set the name of the grid
        self.__name = name
        
        #set the type of grid to be generated
        self.__type = type

        #set the hidden attributes
        self.__children = []
        self.__parent = None
        self.__ref = None
        self.__offsets = None
        self.__mask = None

        #handle each type of grid
        if self.type == GridType.STAND_ALONE:
            #initialize athis as a stand alone grid
            self.__init_as_stand_alone(mask, ref, offsets)

        elif self.type == GridType.ROOT:
            #initialize this as a root grid
            self.__init_as_root(ref, offsets)

        elif self.type == GridType.CHILD:
            #initialize the current grid as a subgrid
            self.__init_as_child(parent, mask)

        else:
            raise RuntimeError("Type enumeration not recognized")
    
    def __del__(self):
        #Grid deconstructor 

        #remove the reference of this grid from the parent
        if self.has_parent:
            self.parent.rem_child(self)

        #delete each child from memory
        for idx in range(0, self.num_children):
            #get the present child
            self.__children[idx].forget_parent()
            #delete the current child
            # del self.__children[0] #seems a little extreme
    
    def __eq__(self,other:_Grid):
        #Checks if two grids are equivalent with one another
        if isinstance(other, Grid):
            return ((self.freqs.size == other.freqs.size) and 
                    (np.all(self.freqs == other.freqs)))
        else:
            return False

    def __init_as_stand_alone(self, mask: np.ndarray, f0: float, delta_f: np.ndarray):
        raise NotImplementedError("Cannot be initialized as a stand-alone grid yet")

    def __init_as_root(self, ref: float, offsets: np.ndarray):
        # Initialize the grid as a root grid
        
        #this will have no parent nodes
        self.__parent = None

        #initialize with no children. 
        self.__children = []
        
        #set the reference frequency 
        self.__ref = ref

        #set the offset frequencies
        if Grid.__is_valid_frequency_array(offsets):
            self.__offsets = offsets
        else:
            raise TypeError("Frequency array is not valid.")
        
        #initialize the mask
        self.__mask = np.full_like(self.__offsets, True, dtype="bool")

    def __init_as_child(self, parent, mask: np.ndarray):
        
        #setup child parent relationship
        if Grid.__is_valid_parent(parent):
            self.__parent = parent

        #create the mask
        if (Grid.__is_valid_mask(mask)) and (mask.size == self.__parent.size):
            self.__mask = mask
        else:
            raise TypeError("Mask is not valid.")

    def add_child(self, new_child):
        if self.__is_valid_child(new_child):
            #add it to the list
            self.__children.append(new_child)
        else:
            raise TypeError("Cannot add submask. Make sure it is the correct size and type.")
    
    def rem_child(self, child_ref):
        #remove the child 
        if isinstance(child_ref, str):
            #get the child by name
            child = self.get_child_by_name(child_ref)
            #handle the case where nothing has been found
            if child is None:
                raise NameError(f"Could not find child with name {child_ref}")
        elif isinstance(child_ref, int):
            #get the child by index
            child = self.get_child_by_index(child_ref)
            #handle the case where None is returned
            if child is None:
                raise IndexError(f"Subgrid could not be found at location {child_ref}")
        elif isinstance(child_ref, Grid):
            #otherwise it is a direct reference to the child
            child = child_ref
        else:
            raise TypeError("child_reference must be a name, index, or Grid")
        
        #now call deconstructor on the child
        del child

    def forget_child(self, child_ref):
        #remove the child 
        if isinstance(child_ref, str):
            #get the child by name
            child = self.get_child_by_name(child_ref)
            #handle the case where nothing has been found
            if child is None:
                raise NameError(f"Could not find child with name {child_ref}")
        elif isinstance(child_ref, int):
            #get the child by index
            child = self.get_child_by_index(child_ref)
            #handle the case where None is returned
            if child is None:
                raise IndexError(f"Subgrid could not be found at location {child_ref}")
        elif isinstance(child_ref, Grid):
            #otherwise it is a direct reference to the child
            child = child_ref
        else:
            raise TypeError("child_reference must be a name, index, or Grid")
        
        #now remove the child from the list of children 
        children_to_forget = [idx for idx, present_child in enumerate(self.__children) if present_child is child]

        #now remove all matches (should only be one but we will use a list to be certain)
        for child_idx in children_to_forget:
            #forget the parent
            self.__children[child_idx].forget_parent()
            #forget the child
            self.__children.remove(child_idx)

    def forget_parent(self):
        # forget the current parent 
        self.__parent = None

    def get_child_by_name(self, name):
        #initialize the reference to none
        ref = None
        #now look for the child
        for idx in range(0,len(self.num_children)):
            if self.__children[idx].__name == name:
                ref = self.__children[idx]
                break
        #return the reference
        return ref
    
    def get_child_by_index(self, idx: int):
        #initialize the reference to none
        if (idx >= 0) and (idx < self.num_children):
            return self.__children[idx]
        else:
            return None
        
    @singledispatchmethod
    def is_compatible(self,x)->bool:
        #This is the default method for is compatible.
        raise NotImplementedError(f"Unsupported type {type(x).__name__} for checking compatibility with {self.name}.")

    @is_compatible.register(_Grid)
    def _(self,x:_Grid, 
          must_be_subgrid:bool=False, 
          must_be_supergrid:bool=False)->bool:
        
        #return true immediately if x is the same as the current grid
        if x is self:
            return True

        #set the return variable at the beginning
        is_x_compatible = self.__shares_root_with(x)
        
        #additional checks
        if must_be_subgrid:
            #TODO: Implement must_be_subgrid flag to is_compatible method
            raise NotImplementedError("must_be_subgrid flag not implemented yet")

        if must_be_supergrid:
            #TODO: Implement must_be_supergrid flag to is_compatible method
            raise NotImplementedError("must_be_supergrid flag not implemented yet")

        #now return whether it is compatible
        return is_x_compatible

    @is_compatible.register(np.ndarray)
    def _(self,x:np.ndarray):
        return self.size == x.size

    def shift_center_to(self, target:float):
        #Shift's center frequency to that specified by the target (in Hz)
        #get the amount of frequency to shift by (positive indicates shift to right)
        
        #throw an error if this grid cannot be shifted
        if not self.has_parent:
            raise RuntimeError("Cannot shift a grid if it is standalone or a root grid")
        
        #get the raw amount to shift by 
        amount = target - self.center_frequency

        #TODO: the next section should autosnap to the parent grid; however, it would be nice if this was automatically reported

        #attempt to shift the grid 
        self.shift(amount)


    def shift(self,amount:int):
        #Shift - Shifts the current grid mask by the amount specified in units of Hz. 
        if not self.has_parent:
            raise RuntimeError("Cannot shift a grid if it is standalone or a root grid.")
        elif not isinstance(amount, (int,float)):
            raise TypeError("Valid arguments for shift are integer or floating point number")

        #we will use the frequency resolution of the parent grid (this may cause undesired behavior if the parent is not uniform)
        parent_resolution = self.parent.frequency_resolution

        #get the amount to wrap around in number of tones (round to the nearest integer) 
        amount = int(np.round(float(amount) / float(parent_resolution)))

        #now roll the mask by the specified amount
        self.roll(amount)
        # self.__mask = np.roll(self.mask, amount)

    def roll(self, amount:int):
        #Roll the current grid by some quantity
        self.__mask = np.roll(self.mask, amount)

    def find_address(self, target:_Grid, 
                     search_subgrids:bool=True, 
                     search_supergrids:bool=True)->list[int]:
        """
        Find the relational address of the target (which may be a grid or grid name) relative to the present grid
        Without any optional arguments this method will search the entire grid tree for the reference by the parent. 
        About the results:
            - 0 indicates that the current grid is the target
            - A negative sign indicates a child index (which child of the prior position to move to)
            - A positive sign integer indicates a parent index (how many parents to move from the current node)
        """
        pres_address = [0]

        if not self is target:
            pres_address = self.__find_address(target, [],
                                search_subgrids=search_subgrids,
                                search_supergrids=search_supergrids, 
                                ignore_child=None)
            
        return pres_address

    def __find_address(self, target, pres_address:list[int],
                       search_subgrids:bool=True, 
                       search_supergrids:bool=True, 
                       ignore_child:Union[_Grid, None]=None, 
                       )->list[int]:
        
        if self is target:
            return pres_address

        if search_subgrids:
            for idx, child in enumerate(self.__children):
                #skip over the current child if it is marked as ignored
                if ignore_child is child:
                    continue
                #set the next address
                next_address = pres_address + [-(idx + 1)]
                #search the present child
                found_address = child.__find_address(target, 
                                                   next_address,
                                                   search_subgrids=True,
                                                   search_supergrids=False)
                #if this was successful
                if len(found_address) > 0:
                    return found_address

        if search_supergrids and self.has_parent:
            #call this method on the parent of the current grid
            if ignore_child is None:
                #we just started searching the supergrids
                #if ignore_child is still None
                next_address=pres_address + [1] #moving up to next parent
            else:
                next_address = pres_address
                next_address[-1] = next_address[-1] + 1

            #update the ignore child
            ignore_child = self

            #Call the find address method on the parent
            found_address = self.__parent.__find_address(target, 
                                                    next_address, 
                                                    search_subgrids=search_subgrids,
                                                    search_supergrids=True,
                                                    ignore_child=ignore_child)
            #now return the value if 
            if len(found_address) > 0:
                return found_address

        #return an empty list if nothing was found
        return []

    def cast(self, data: np.ndarray, target_grid, off_grid_vals=float(0), dtype=None):
        """
        Method for casting data between two grids. 
        This is a very inefficient method for doing this but seems relatively 
        straight forward. It shouldn't be a problem for arrays of moderate size 
        and with limited depth. This should be the most general approach.
        """

        #TODO: This is inefficient when doing multiple successive data casts, i.e. both a and b waves
        #Consider allowing a list or tuple to pass through for multiple data sets of the same
        #size.

        #step-1: If the two grids, i.e. self and target_grid, are the same. Just return the data
        if target_grid is self:
            return data

        #step0: perform some checks
        #make sure that the data is valid on the source grid (otherwise what's the point)
        if not self.is_compatible(data):
            raise ValueError("Grid cast data not compatible with source grid.")

        #target grid must be a valid grid
        if not isinstance(target_grid, Grid):
            raise ValueError("Target grid must be an instance of the source grid.")
        
        #get the data type of the new grid to use
        if dtype is None:
            #use the datatype of the object
            dtype = data.dtype
        elif isinstance(dtype, np.dtype):
            #nothing to do
            pass
        elif isinstance(dtype, str):
            #cast to the requested datatype
            dtype = np.dtype(dtype)
        else:
            #throw an error
            raise ValueError("Proposed datatype is not valid.")
        
        #make sure we are all working on the same source grid.
        if not self.__shares_root_with(target_grid):
            raise RuntimeError("Target grid and source grid do not share a root grid.")
        
        #step1: cast both grids to the root grid
        source_grid = self.cast_to_root()
        target_grid = target_grid.cast_to_root()

        #step2: setup the data at the root grid
        #initialize the data at root to the off-grid values
        data_at_root = np.full_like(source_grid, off_grid_vals, dtype=dtype)
        #now set the data cast from the source to the root
        data_at_root[source_grid] = data
        #step3: return the data at the root as it appears to the target grid
        return data_at_root[target_grid]

    def cast_to_parent_grid(self, pres_grid=None):
        #if the present grid is None, create a new grid with a full array of true values 
        if pres_grid is None:
            pres_grid = np.full(self.size, True)
        
        #get the indices of the next grid to retain
        next_grid_idxs = np.where(self.__mask)
        next_grid_idxs = next_grid_idxs[0]              #get from tuple 
        next_grid_idxs = next_grid_idxs[pres_grid]      #apply present grid to indices

        #cast to the parent grid using the current grid's mask
        next_grid = np.full(self.__parent.size, False)
        next_grid[next_grid_idxs] = True

        #return the next grid
        return next_grid

    def cast_to_root(self, pres_grid = None):
        #if the present grid is None, create a new grid with a full array of true values 
        if pres_grid is None:
            pres_grid = np.full(self.size, True)
        
        if not self.has_parent:
            #if it doesn't have a parent, return the present grid
            return pres_grid
        else:
            #cast to the parent grid
            next_grid = self.cast_to_parent_grid(pres_grid=pres_grid)
            #now move to the parent and repeat until we reach the root
            return self.__parent.cast_to_root(pres_grid = next_grid)
    
    def cast_index(self, to_grid, about_center:bool=False):
        #cast the indexing of the current grid onto another grid
        
        #generate the indexing of the current grid
        pres_idx = self.index(about_center=about_center)

        #cast the current indexing to the target grid
        cast_idx = self.cast(pres_idx, to_grid, None, dtype="object")

        #indices that failed to map from one grid to the other will be of type None
        #we should remove them here
        valid_idxs = np.where( np.logical_not(cast_idx == None))
        valid_idxs = valid_idxs[0]

        #only keep parts of the array where the indexing is valid
        cast_idx = cast_idx[valid_idxs]

        #now cast to type int
        return cast_idx.astype(np.dtype("int64"))

    def index(self, about_center:bool=False, round_down:bool=True):
        #Generate index for the present grid
        #TODO: make this about the reference frequency rather than the frequency

        #generate the initial indices
        idxs = np.arange(self.size)

        #generate indexing about the center of the grid rather than the left side
        if about_center:
            #get the central index
            cntr = np.mean(idxs)
            #handle the case where the number of indices is even
            if (idxs.size % 2 == 0):
                if round_down: 
                    #round the central index down
                    cntr = np.floor(cntr)
                else:
                    #round the central index up
                    cntr = np.ceil(cntr)
            #apply the offset
            idxs = idxs - cntr
            #convert to type integer (not float)
            idxs = idxs.astype(np.dtype("int64"))
        
        #return the indices
        return idxs
    
    def zeros_like(self, dtype=None)->np.ndarray:
        #Return the data as zeros
        return np.zeros_like(self.freqs, dtype=dtype)
    
    def ones_like(self, dtype=None)->np.ndarray:
        #Return the data as ones
        return np.ones_like(self.freqs, detype=dtype)
    
    def full_like(self, fill_value, dtype=None):
        #Create a full-like array array of objects the size of this grid
        return np.full_like(self.freqs, fill_value, dtype=dtype)

    #for the grid type
    @property
    def type(self):
        return self.__type
    
    @property
    def name(self):
        return self.__name
    
    @property 
    def num_children(self):
        if self.__type == GridType.STAND_ALONE:
            return 0
        else:
            return len(self.__children)
        
    @property 
    def has_children(self):
        return not (self.num_children == 0)
    
    @property
    def has_parent(self):
        return not (self.__parent is None)
    
    @property
    def parent(self):
        return self.__parent
    
    @parent.setter
    def parent(self, new_parent):
        if self.has_parent:
            raise RuntimeError("Grid already has a parent. Please call forget parent first.")
        

    def adopt_parent(self, new_parent):

        #Make sure that the new entry makes sense
        if self.has_parent:
            raise RuntimeError("Grid already has a parent. Please call forget parent first.")
        elif not isinstance(new_parent,Grid):
            raise TypeError("New parent grid must be a Grid object.")
        elif not self.mask.size == new_parent.size:
            raise ValueError(f"Present mask size {self.mask.size} must match that of the parent grid size {new_parent.size}")

        #if everythin passed, then we can set the new parent
        self.__parent  = new_parent

    @property
    def freqs(self):
        """
        The return value of this property depends on the type of grid calling it.
        """
        if self.__type == GridType.STAND_ALONE:
            #generate the absolute frequencies 
            freqs = self.__ref + self.__offsets
            #return the frequencies where mask is true
            return freqs[self.__mask]
        
        elif self.__type == GridType.ROOT:
            #simply return all frequencies (mask is true everywhere)
            return self.__ref + self.__offsets
        
        elif self.__type == GridType.CHILD:
            #return the mask applied to the Parent's frequencies
            return self.__parent.freqs[self.__mask]
    
    @property
    def ref_freq(self):
        if self.__type == GridType.CHILD: 
            return self.__parent.ref_freq
        else:
            return self.__ref
    
    @property 
    def offset_freqs(self):
        if self.__type == GridType.STAND_ALONE:
            #generate the absolute frequencies 
            freqs = self.__offsets
            #return the frequencies where mask is true
            return freqs[self.__mask]
        
        elif self.__type == GridType.ROOT:
            #simply return all frequencies (mask is true everywhere)
            return self.__offsets
        
        elif self.__type == GridType.CHILD:
            #return the mask applied to the Parent's frequencies
            return self.__parent.offset_freqs[self.__mask]

    @property
    def size(self)->int:
        #returns the size of the grid (or the data that it exists on)
        return int(np.sum(self.__mask))
    
    @property
    def mask_size(self):
        #returns the size of the mask
        return self.__mask.size
    
    @property
    def root(self):
        if not self.has_parent:
            #stand alone and roots are their own root
            return self
        elif self.__type == GridType.CHILD: 
            #get the root of the parent
            return self.__parent.root

    @property
    def mask(self):
        return self.__mask
    
    @property
    def frequency_resolution(self):
        #Returns the frequency resolution of the current grid 
        f = self.freqs
        #get an array of frequency steps in the grid
        f_step = f[1:-1] - f[0:-2]
        #the frequency resolution will be the minimum of all f_steps
        return np.min(f_step)

    @property 
    def bandwidth(self):
        #Returns the bandwidth of the current grid
        return self.freqs[-1] - self.freqs[0]
    
    @property 
    def period(self):
        #Returns the expected period of any signal on the grid

        #The period of the grid is the same as 1 over the minimum step in frequency
        return 1 / self.frequency_resolution
    
    @property 
    def center_frequency(self):
        #Returns the frequency (on the root grid) that is halfway between the extenses of this grid's frequency
        grid_freqs = self.freqs
        root_freqs = self.root.freqs
        #get the mean frequency
        mean_freq = (grid_freqs[0] + grid_freqs[-1])/2
        #return the closest on-grid frequency that matches
        return root_freqs[np.argmin(np.abs(mean_freq-root_freqs))]

    """
    Other Code Not Implemented Yet
    """

    # def is_compatible(self, x)->bool:
    #     #the size of the grid and the size of the data must be equal
    #     if isinstance(x, np.ndarray):
    #         return self.size == x.size
    #     elif isinstance(x, Grid):
    #         return self.__shares_root_with(x)
    #     else:
    #         raise TypeError("Unrecognized type for is_compatible check on present grid")

    # @singledispatchmethod
    # def find_offspring_address(self, x)->list[int]:
    #     raise NotImplementedError(f"Unsupported type {type(x).__name__} for finding offspring address relative to {self.name}.")

    # @find_offspring_address.register(__Grid)
    # def _(self, other_grid:__Grid, ignore:Union[__Grid,None])->list[int]:
    #     #Find the indices of the children 
        
    #     #initialize the list of return indices
    #     address_of_child = []

    #     #search through the children
    #     if self.num_children == 0:
    #         address_of_child = []
    #     else:
    #         for idx, child in enumerate(self.__children):
    #             if child is other_grid:
    #                 address_of_child = [idx]
    #                 break
    #             else:
    #                 #determine if the child has the offspring
    #                 if ignore is child:
    #                     continue
    #                 #otherwise search the current children for the address
    #                 chld_list = child.find_offspring_address(other_grid)
    #                 #if the list isn't empty we should return the current index 
    #                 #with the child's list appended to it
    #                 if len(chld_list) > 0:
    #                     #set the current index to the index of child
    #                     address_of_child =  [idx] + chld_list
    #                     #break from the loop
    #                     break
        
    #     #return the result
    #     return address_of_child
    
    # @find_offspring_address.register(str)
    # def _(self, offspring_name:str, ignore:Union[__Grid,None])->list[int]:
    #     #Find the indices of the children 
        
    #     #initialize the list of return indices
    #     address_of_child = []

    #     #search through the children
    #     if self.num_children == 0:
    #         address_of_child = []
    #     else:
    #         for idx, child in enumerate(self.__children):
    #             if child.name == offspring_name:
    #                 address_of_child = [idx]
    #                 break
    #             else:
    #                 #determine if the child has the offspring
    #                 if ignore is child:
    #                     continue
    #                 #determine if the child has the offspring
    #                 chld_list = child.find_offspring_address(offspring_name)
    #                 #if the list isn't empty we should return the current index 
    #                 #with the child's list appended to it
    #                 if len(chld_list) > 0:
    #                     #set the current index to the index of child
    #                     address_of_child =  [idx] + chld_list
    #                     #break from the loop
    #                     break
        
    #     #return the result
    #     return address_of_child

    # @singledispatchmethod
    # def find_predecessor_address(self, x)->int:
    #     raise NotImplementedError(f"Unsupported type {type(x).__name__} for finding predecessor address relative to {self.name}.")
    
    # @find_predecessor_address.register(__Grid)
    # def _(self, other_grid:__Grid, pres_distance_up:int=0)->int:
    #     if self is other_grid:
    #         #return the current distance upward
    #         return pres_distance_up
    #     elif not self.has_parent:
    #         #we are at the root
    #         return -1
    #     else:
    #         return self.__parent.find_predecessor_address(other_grid, pres_distance_up= pres_distance_up + 1)
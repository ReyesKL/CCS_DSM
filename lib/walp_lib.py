#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import xarray as xr
import cf_xarray as cfxr
from pathlib import Path
import copy


# A library of some basic functions to use for WALP
class DataSaver:

    def __init__(self,
                 path,
                 logger=None,
                 coords =None,
                 meas_attrs=None):
        
        ## Initialize the DataSaver object
        # assert os.path.exists(path), "Path to target does not exist. Cannot save file."
        self._path = Path(path)

        #add the logger
        self.logger = logger

        #initialize attributes to save 
        self.coords = coords
        self.keys = self.coords.keys()
        self.coords_no_f = copy.deepcopy(self.coords)
        del self.coords_no_f["frequency"]
        self.keys_no_f = self.coords_no_f.keys()
        self.meas_attrs = meas_attrs

        self.coords_f = {"frequency": self.coords["frequency"]}
        self.keys_f = ["frequency"]

        # Initialize the start and stop times
        self.startTime = datetime.now()
        self.stopTime  = self.startTime
        
        #properties that need to be set
        # For the a and b ways
        self._a1_dat                                    = None
        self._a2_dat                                    = None 
        self._b1_dat                                    = None
        self._b2_dat                                    = None

        # reference signal
        self._s0_dat = None
        
        # Gate properties 
        self._gate_voltage_dat                          = None
        self._gate_current_driven_dat                   = None
        # self._gate_current_quiescent_before_sweep_dat   = None
        # self._gate_current_quiescent_after_sweep_dat    = None

        # Drain properties
        self._drain_voltage_dat                         = None 
        self._drain_current_driven_dat                  = None
        # self._drain_current_quiescent_before_sweep_dat  = None
        # self._drain_current_quiescent_after_sweep_dat   = None

        # For the excitation grid
        self._excitation_grid                           = None
    
    #getter function for the 
    @property 
    def path(self):
        return self._path
    
    @property 
    def a1_dat(self):
        return self._a1_dat
    
    @property 
    def a2_dat(self):
        return self._a2_dat
    
    @property 
    def b1_dat(self):
        return self._b1_dat
    
    @property 
    def b2_dat(self):
        return self._b2_dat

    @property
    def s0_dat(self):
        return self._s0_dat
    
    @property
    def gate_voltage_dat(self):
        return self._gate_voltage_dat
    
    @property 
    def gate_current_driven_dat(self):
        return self._gate_current_driven_dat

    # @property
    # def gate_current_quiescent_before_sweep_dat(self):
    #     return self._gate_current_quiescent_before_sweep_dat
    #
    # @property
    # def gate_current_quiescent_after_sweep_dat(self):
    #     return self._gate_current_quiescent_after_sweep_dat

    @property
    def drain_voltage_dat(self):
        return self._drain_voltage_dat
    
    @property
    def drain_current_driven_dat(self):
        return self._drain_current_driven_dat

    # @property
    # def drain_current_quiescent_before_sweep_dat(self):
    #     return self._drain_current_quiescent_before_sweep_dat
    #
    # @property
    # def drain_current_quiescent_after_sweep_dat(self):
    #     return self._drain_current_quiescent_after_sweep_dat

    @property
    def excitation_grid(self):
        return self._excitation_grid

    #setter functions 
    # path
    @path.setter
    def path(self,newPath):
        self._path = Path(newPath)
    
    # a1_dat
    @a1_dat.setter
    def a1_dat(self,dat):
        self._a1_dat = xr.DataArray(data=dat,
                                    coords=self.coords,
                                    dims=self.keys)
    
    # a2_dat
    @a2_dat.setter
    def a2_dat(self,dat):
        self._a2_dat = xr.DataArray(data=dat,
                                    coords=self.coords,
                                    dims=self.keys
                                    )

    # b1_dat
    @b1_dat.setter
    def b1_dat(self,dat):
        self._b1_dat = xr.DataArray(data=dat,
                                    coords=self.coords,
                                    dims=self.keys
                                    )
    
    # b2_dat 
    @b2_dat.setter
    def b2_dat(self,dat):
        self._b2_dat = xr.DataArray(data=dat,
                                    coords=self.coords,
                                    dims=self.keys
                                    )

    @s0_dat.setter
    def s0_dat(self,dat):
        self._s0_dat = xr.DataArray(data=dat, coords=self.coords_f, dims=self.keys_f)

    # gate_voltage_dat
    @gate_voltage_dat.setter
    def gate_voltage_dat(self,dat, dim_names):
        self._gate_voltage_dat = xr.DataArray(data=dat,
                                                coords=self.coords,
                                              dims=self.keys
                                              )
    
    # gate_current_driven_dat
    @gate_current_driven_dat.setter
    def gate_current_driven_dat(self,dat):
        self._gate_curren_driven_dat = xr.DataArray(data=dat,
                                                    coords=self.coords_no_f,
                                                    dims=self.keys_no_f
                                                        )
    

    # drain_voltage_dat
    @drain_voltage_dat.setter
    def drain_voltage_dat(self,dat):
        self._drain_voltage_dat = xr.DataArray(data=dat,
                                               coords=self.coords_no_f,
                                               dims=self.keys_no_f
                                               )
    
    # drain_current_driven_dat
    @drain_current_driven_dat.setter
    def drain_current_driven_dat(self,dat):
        self._drain_current_driven_dat = xr.DataArray(data=dat, 
                                                      coords=self.coords_no_f,
                                                      dims=self.keys_no_f
                                                      )
    
      # excitation_grid
    @excitation_grid.setter
    def excitation_grid(self,dat):
        excitation_grid = xr.DataArray(data=dat,
                                        coords=[self.freqs],
                                        dims=["Frequency"])
        
    # update the start and stop times
    def tic(self):
        self.startTime = datetime.now()
    
    def toc(self):
        self.stopTime  = datetime.now()

    def saveData(self, meta_str, successful=False):
        #pull the provided measurement attributes
        meas_attrs = self.meas_attrs

        #calculate the measurement time
        measTime  = self.stopTime - self.startTime
        
        #save the start and stop times
        meas_attrs["StartDate"]             = self.startTime.strftime("%d/%m/%Y %H:%M:%S")
        meas_attrs["EndDate"]               = self.stopTime.strftime("%d/%m/%Y %H:%M:%S")
        meas_attrs["Duration"]              = measTime.seconds

        #measAttrs["EndDate"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        #measAttrs["EndDate"]
        
        #now save whether or not the measurement was successful
        meas_attrs["CompletedSuccessfully"] = successful

        data_dict = dict(a1=self.a1_dat, 
                        b1=self.b1_dat, 
                        a2=self.a2_dat, 
                        b2=self.b2_dat,
                        s0=self.s0_dat,
                        Vgg_measured=self.gate_voltage_dat, 
                        Vdd_measured=self.drain_voltage_dat,
                        Igg_driven=self.gate_current_driven_dat, 
                        Idd_driven=self.drain_current_driven_dat,
                        # Igg_quiescent_before=self.gate_current_quiescent_before_sweep_dat,
                        # Igg_quiescent_after=self.gate_current_quiescent_after_sweep_dat,
                        # Idd_quiescent_before=self.drain_current_quiescent_before_sweep_dat,
                        # Idd_quiescent_after=self.drain_current_quiescent_after_sweep_dat,
                        ExcitationGrid=self.excitation_grid)

        sweep_dat = xr.Dataset(data_dict, attrs=meas_attrs)
        encoded = cfxr.encode_multi_index_as_compress(sweep_dat, idxnames=["load_gamma", "source_gamma"])


        fname = self._path.with_name(str(self.path.stem) + meta_str + str(self.path.suffix))
        
        encoded.to_netcdf(fname, engine='h5netcdf', invalid_netcdf=True)
        if self.logger is not None:
            self.logger.info("Data saved successfully")
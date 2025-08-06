import numpy as np
import xarray as xr


class signal:

    def __init__(self, num_tones, phases=None, amplitudes=None, par=np.nan, tone_spacing=np.nan, f0=np.nan,
                 grid_slots=np.nan):
        self.num_tones = num_tones
        if amplitudes is None:
            self.amplitudes = np.ones(num_tones)
        else:
            self.amplitudes = amplitudes
        self.phases = phases

        self.__par = par
        self.__tone_spacing = tone_spacing
        self.__f0 = f0

        # todo
        self.__absolute_amplitudes = False
        # when this is toggled true, the VST signal player will set that option

        if np.any(np.isnan(grid_slots)):
            self.__grid_slots = np.empty(num_tones)
            self.__grid_slots[:] = np.NaN
            # todo this may not work.
        else:
            self.__grid_slots = grid_slots

        # todo we need an excited_slots array or similar
        #  we can't keep counting on amps to always be 1 for an excited tone
        #  also may be good to store tone spacing here

        # todo let's also create save/load methods so we can effectively pickle a signal
        #  no need to keep translating between our xarray data and the signal class


    @classmethod
    def random_phase_equal_amplitude(cls, num_tones):
        phases = np.random.rand((num_tones)) * 2 * np.pi
        amplitudes = np.ones(num_tones)
        return cls(num_tones, phases, amplitudes)

    @classmethod
    def random_phase_random_amplitude(cls, num_tones):
        phases = np.random.rand(num_tones) * 2 * np.pi
        amplitudes = np.random.rand(num_tones)
        return cls(num_tones, phases, amplitudes)

    @classmethod
    def schroeder_phase_equal_amplitude(cls, num_tones):
        phases = schroeder_phase(num_tones)
        amplitudes = np.ones(num_tones)
        return cls(num_tones, phases, amplitudes)

    @classmethod
    def equal_phase_equal_amplitude(cls, num_tones):
        phases = np.zeros(num_tones)
        amplitudes = np.ones(num_tones)
        return cls(num_tones, phases, amplitudes)

    @classmethod
    def from_h5(cls, fname):
        data = xr.open_dataset(fname, engine='h5netcdf')
        grid_slots = data["grid_slots"].data
        pars = data["pars"].data
        tone_spacing = data.Tone_Spacing
        grid_slots -= np.min(grid_slots)  # this effectively takes you down to bb if not already there.
        # amps = np.zeros(np.max(grid_slots) + 1)
        # amps[grid_slots] = 1
        amps = np.ones_like(grid_slots)
        # grid slots can be thought of as indices,
        # so we need to make the holding array 1 larger so all indices are valid



        phases = data.phases.data
        signals = []
        for i_sig, sig in enumerate(data["Test_Signal"]):
            # num_tones = len(amps)
            # todo this is really not correct.
            # we will need to be very careful about how many excited slots
            # the signal occupies and how many are actually excited
            # can we just make everything general so it doesn't matter?
            num_tones = np.count_nonzero(amps[i_sig])
            # better
            signals.append(cls(num_tones, phases[i_sig, :], amps[i_sig], par=pars[i_sig],
                               grid_slots=grid_slots[i_sig], tone_spacing=tone_spacing))
        return signals
        # this may not work. It's also predicated on the assumption that
        # signals within a file have the same amplitudes/ grid locs
        # and different phases.

    def to_h5(self, fname):
        phase_dat = xr.DataArray(data=self.phases,
                                 dims=["Amplitudes"], coords=[self.amplitudes])

        excitation_coords = np.linspace(0, len(self.__grid_slots), len(self.__grid_slots))
        grid_dat = xr.DataArray(data=self.__par,
                                dims=["Excitation_Slots"], coords=[excitation_coords])

        data_dict = dict(phases=phase_dat, excitation_grid=grid_dat)

        meas_attrs = {"PAR": self.__par, "F0": self.__f0, "Tone_Spacing": self.__tone_spacing}
        sweep_dat = xr.Dataset(data_dict, attrs=meas_attrs)

        sweep_dat.to_netcdf(fname, engine='h5netcdf')  # , invalid_netcdf=True)

        # todo do we also want to save the grid and the tone spacing
        #  also need the par, par and tone_spacing can be saved as attributes.
        #  the coordinates should really be grid slots

    @staticmethod
    def schroeder_phase(n):
        # n is the number of tones
        k = np.arange(0, n + 1, 1)
        phases = (-k * (k - 1) * np.pi) / n
        return phases

    def calculate_par(self):
        # take ifft,  calculate par
        pass

    def set_par(self, par):
        if par > 0:
            self.__par = par
        else:
            raise Warning("PAR cannot be less than Zer0")

    def get_par(self):
        return self.__par

    def set_grid_slots(self, grid_slots):
        if len(grid_slots) == len(self.amplitudes):
            self.__grid_slots = grid_slots
        else:
            raise Warning("Grid slots is not of correct size")

    def get_grid_slots(self):
        return self.__grid_slots

    def set_tone_spacing(self, tone_spacing):
        if tone_spacing > 0:
            self.__tone_spacing = tone_spacing
        else:
            raise Warning("Tone spacing must be greater than zero")

    def get_tone_spacing(self):
        return self.__tone_spacing
    
    def fix_busted_notch(self):
        """A simple function to widen and oddify the notch size
           This should not be widely or generally used"""
        dx = np.diff(self.__grid_slots)  # calculate the difference between slots
        notch_loc = np.argmax(dx) + 1  # find the largest, corresponding to the notch location
        self.__grid_slots[notch_loc:] += 21  # move the upper part of the signal up
        # we already have a 10 tone notch so this will make the notch 31 tones wide
        # it is likely not safe to move the lower part down instead because then the idxs will start below zero
        # effectively we are just widening the notch by 21 tones so it is both wider (~10% of 300-tone) and odd.

    def set_f0(self, f0):
        if f0 >= 0:
            self.__f0 = f0

        else:
            raise Warning("Center frequency must be greater than or equal to zero")

    def get_f0(self):
        return self.__f0

    def scale_amplitudes_to_one(self):
        # will be used in conjunction with "interpret_multitone_amplitudes_as_absolute" =True
        pass

    def drop_one_in_n_randomly(self, n):
        # thanks chatgpt
        # Calculate the number of groups and the remaining elements
        num_groups, remainder = divmod(len(self.amplitudes), n)

        # Loop through the groups and select elements
        for i in range(num_groups):
            group_start = i * n
            group_end = group_start + n
            selected_index = np.random.randint(group_start, group_end)
            self.amplitudes[selected_index] = 0

        # Handle the boundary condition
        if remainder > 0:
            last_group_start = num_groups * n
            last_group_end = num_groups * n + remainder
            selected_index = np.random.randint(last_group_start, last_group_end)
            self.amplitudes[selected_index] = 0

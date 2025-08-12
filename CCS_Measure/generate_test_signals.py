"""
Title:  Generate test signals
Author: Paul Flaten
Discription: Simple script for generating test signals.
"""

# Import Statements
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.vst_util_lib as util
from lib.waveform_generator import Multitone_Waveform_Generator as AWG
from lib.active_tuner_grid_lib import Grid, GridGenerator, FrequencyGridSource



#Script Settings
# For the measurement grid
root_grid  = {"center_frequency": 4.5e9,
              "grid_step": 1e6, 
              "points": 991}

# Signal settings
output_directory    = os.path.join(os.getcwd(), "CCS_Measure\signals")
name                = "Signal_10MHz_4.25GHz"
bandwidth           = 10e6
notch_frac          = 0.1
target_par          = [3,4,5,6,7,8]
carrier_frequency   = 4.25e9
plot_signal         = True

#Create the measurement grid
grid_source = FrequencyGridSource(root_grid["center_frequency"], size=root_grid["points"], step=root_grid["grid_step"])
grid_gen    = GridGenerator(grid_source)
measurement_grid = Grid.generate("Measurement", using=grid_gen)

#Build the signal generator
gen = AWG(name, measurement_grid, bandwidth, notch_frac_width=notch_frac, center_frequency=carrier_frequency,trials=1001)

#print the range of PAR values to the terminal
print(f"PAR values for the current generator: {gen.min_par:.2f}dB to {gen.max_par:.2f}dB")

for par in target_par:
    #Get the closest signal with the target PAR
    sig,par_found = gen.get_signal_with_par(par)

    #Save the target signal 
    sig.save(os.path.join(output_directory, f"{name}_{par_found:.1f}dB_PAR.h5"))

    #Get the envelope of the signal
    A0, t = sig.complex_baseband_time_domain()

    #Create a DataFrame
    df = pd.DataFrame({"Time": t, "Signal": np.abs(A0)})

    #Save the datafrom to csv file
    df.to_csv(os.path.join(output_directory, f"{name}_{par_found:.1f}dB_PAR.csv"), index=False)

    if plot_signal:
        plt.plot(t, np.abs(A0))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

import numpy as np
import scipy



class DsmAligner:
    def __init__(self, scope, log, rf_source, pa_chn, dsm_chn):
        self.scope = scope
        self.log = log
        self.rf_source = rf_source
        self.pa_chn = pa_chn
        self.dsm_chn = dsm_chn
        # will need the RF source and o-scope objects



    def align(self, ):

        aligned = False

        while not aligned:
            # measure both waveforms
            t, pa_wvfm = self.scope.get_td_data(self.pa_chn)
            _, dsm_wvfm = self.scope.get_td_data(self.dsm_chn)
            # get envelope of rf waveform
            pa_env_wvfm = scipy.signal.hilbert(pa_wvfm)

            # cross correlate the wave   forms and find the delay
            corr = scipy.signal.correlate(pa_env_wvfm, dsm_wvfm)
            lags = scipy.signal.correlation_lags(len(pa_env_wvfm), len(dsm_wvfm))
            lag = lags[np.argmax(np.abs(corr))]

            # apply the delay to the rf source

            # repeat until the waveforms are aligned
            # need to define some convergence criteria
            # need to possibly upsample/filter the waveforms


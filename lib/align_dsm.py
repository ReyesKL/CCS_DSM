import numpy as np
import scipy
import scipy.fft as fft
import matplotlib.pyplot as plt
from lib.RKL_TOOLS import find_nearest_idx, normalize


def filter_and_decimate(t, v):

    dt = t[1] - t[0]
    t_max = np.max(t)
    t_min = np.min(t)
    v_f = fft.fftshift(fft.fft(v))
    f = fft.fftshift(fft.fftfreq(v.size, dt))

    f_max_idx = find_nearest_idx(f, 100e6)
    v_f[f_max_idx:] = 0
    v = fft.ifft(fft.ifftshift(v_f))
    v = scipy.signal.decimate(v, 10)
    t = np.linspace(t_min, t_max, v.size)

    return t, v


class DsmAligner:
    def __init__(self, scope, log, rf_source, signal_period, pa_chn, dsm_chn):
        self.scope = scope
        self.log = log
        self.rf_source = rf_source
        self.pa_chn = pa_chn
        self.dsm_chn = dsm_chn
        self.signal_period = signal_period
        self.rf_source.Delay = 0.
        # #############################
        # #for testing only
        # self.rf_source.Delay = 0.19575e-6
        # #for testing only
        # #############################



    def align(self, debug=False, atol=1e-9, n_its=10 ):

        aligned = False
        i = 0
        while not aligned and i<n_its:
            # measure both waveforms
            self.scope.auto_scale(self.pa_chn)
            self.scope.auto_scale(self.dsm_chn)
            t, pa_wvfm = self.scope.get_td_data(self.pa_chn, update_view=False)
            pa_wvfm = self.scope.de_embed_td_data(self.pa_chn, t, pa_wvfm)

            t, dsm_wvfm = self.scope.get_td_data(self.dsm_chn, update_view=False, rerun=False)

            # get envelope of rf waveform
            pa_env_wvfm = np.abs(scipy.signal.hilbert((pa_wvfm)))
            pa_env_wvfm = normalize(pa_env_wvfm, norm_type="min-max")

            # #############################
            # # for testing only
            # dsm_wvfm = np.abs(scipy.signal.hilbert(np.abs(dsm_wvfm)))
            # dsm_wvfm = dsm_wvfm / np.max(dsm_wvfm)
            # # for testing only
            # #############################


            t, pa_env_wvfm = filter_and_decimate(t, pa_env_wvfm)
            t, dsm_wvfm = filter_and_decimate(t, dsm_wvfm)
            dsm_wvfm = normalize(dsm_wvfm, norm_type="min-max")


            # cross correlate the wave forms and find the delay
            corr = scipy.signal.correlate(pa_env_wvfm, dsm_wvfm)
            lags = scipy.signal.correlation_lags(len(pa_env_wvfm), len(dsm_wvfm))
            lag = lags[np.argmax(np.abs(corr))]
            t_delay = np.roll(t,lag)[0]
            t_delay = t_delay % self.signal_period
            if t_delay > self.signal_period/2:
                t_delay -= self.signal_period
            self.log.info(f"Found delay of {t_delay*1e9:.2f} ns")

            # apply the delay to the rf source
            self.rf_source.Delay += t_delay
            self.rf_source.Delay = self.rf_source.Delay % self.signal_period

            if np.isclose(t_delay, 0, atol=atol):
                aligned = True

            i += 1
        if aligned:
            self.log.info(f"DSM waveforms aligned in {i} its")
            self.log.info(f"DSM delay is {self.rf_source.Delay*1e9:.2f} ns")
            if debug:
                plt.ioff()
                fig, ax = plt.subplots()
                ax.plot(t, pa_env_wvfm, label="RF envelope")
                ax.plot(t, dsm_wvfm, label="DSM envelope")
                ax.legend(loc="upper right")
                ax.grid()
                plt.show()
        else:
            self.log.error(f"DSM waveforms did not align in {i} its")

            # repeat until the waveforms are aligned
            # need to define some convergence criteria
            # need to possibly upsample/filter the waveforms


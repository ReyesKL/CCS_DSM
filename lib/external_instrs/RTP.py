import time

import numpy as np
import scipy.io as sio
from pathlib import Path, WindowsPath
from rich.progress import Progress
from lib.external_instrs import ScopeViewer
import skrf as rf

class RTP:

    def __init__(self, resource_manager, ip_addr, log, name):
        self.rm = resource_manager
        self.rtp = self.rm.open_resource(ip_addr)
        self.rtp.timeout = 200000
        self.log = log
        self.name = name
        self.log.info(f"{self.name}: Initialized")
        v_scale_min = 20e-3
        v_scale_max = 10
        # num_steps = int(v_scale_max / v_scale_min)
        self.ranges = np.arange(v_scale_min,v_scale_max+v_scale_min,v_scale_min)
        self.scope_view = ScopeViewer.ScopeViewer()
        self._update_viewer_time()
        self.fixtures = [None, None, None, None]
        self.port_on_dut = [1, 1, 1, 1]

    def get_td_data(self,channel, rerun=True, update_view=True):
        self.log.info(f"{self.name}chn:{channel}: Measuring time domain waveform")
        channel = int(channel)

        if rerun:
            self.rtp.write("RUNS")
        self.rtp.write(f"FORM:DATA REAL,32")
        dat = self.rtp.query_binary_values(f"CHAN{channel}:WAV1:DATA?")
        # dat = self.rtp.query_ascii_values(f"CHAN{channel}:WAV1:DATA?")

        # dat = dat.split(',')
        # dat = list(float(x) for x in dat)
        header = self.rtp.query(f"CHAN{channel}:WAV1:DATA:HEAD?").strip()
        header = header.split(',')
        t0 = float(header[0])
        t1 = float(header[1])
        ns = int(header[2])
        t = np.linspace(0, t1-t0, ns)
        if update_view:
            # self.scope_view.set_time_base(t)
            self.scope_view.update_trace(channel, dat)

        return t, dat
    
    def de_embedd_td_data(self, channel, t, dat):
        

        if self.fixtures[channel]:
            #get the network parameters of the corresponding fixture
            net = self.fixtures[channel]

            #get the time step
            dt = t[1]-t[0]

            #Step 1: convert data to frequency domain
            V = np.fft.fft(dat)
            f = np.fft.fftfreq(dat.size, dt)

            #Step 2: truncate the data to the network frequencies 
            network_frequencies = net.f
            min_freq = np.min(network_frequencies); max_freq = np.max(network_frequencies)
            keep = np.logical_and(f >= min_freq, f <= max_freq)

            #truncate the data to the values inside of the desired frequency range 
            Vx = V[keep]; fx = f[keep]
            
            #Step 3: now perform the de-embedding 
            net = net.interpolate(fx)
            if self.port_on_dut[channel]==1: #port 1 is on the DUT
                Vx = Vx * (1 + net.z0 * net.s11) / (net.z0 * net.s21)
            elif self.port_on_dut[channel]==2: 
                Vx = Vx * (1 + net.z0 * net.s22) / (net.z0 * net.s12)
            else:
                raise ValueError("Fixture port index on dut must be 1 or 2")

            #reset the voltage array
            V = np.zeros_like(V, dtype="complex")
            V[keep] = Vx[:]

            #Step 4: finally take the ifft (real voltages should only result in real voltages)
            dat = np.real(np.fft.ifft(V))

        #return time domain data
        return dat

    def auto_scale(self,chn):
        self.set_chn_scale(chn,1)
        old_scale = np.nan
        new_scale = 0.5
        with Progress() as progress:
            task = progress.add_task(f"Auto-scaling channel {chn}", total=None)
            while not np.isclose(old_scale,new_scale, rtol=1e-1,atol=1e-3):
                t , v = self.get_td_data(chn)
                # v_mn = np.mean(v)
                v_pk = np.max(v)
                v_min = np.min(v)
                v_rang = v_pk - v_min
                # v_rang = np.nanmax([v_rang, old_scale])

                old_scale = new_scale
                # new_scale = 5 * (v_rang/10)
                new_scale = v_rang / 2
                # self.set_offset(v_mn, chn)
                self.set_chn_scale(chn,new_scale)
                time.sleep(0.1)
            progress.remove_task(task)
        # return (t, d)
    def set_chn_scale(self,chn, scale):
        self.rtp.write(f"CHAN{chn}:SCAL {scale}")

    def set_offset(self, lvl, chn):
        self.rtp.write(f"CHAN{chn}:OFFS {lvl}")

    def set_acq_time(self, acq_time):
        self.rtp.write(f"TIM:RANG {acq_time}")
        self._update_viewer_time()

    def set_sample_rate(self, sample_rate):
        self.log.info(f"{self.name}: Setting sample rate to {sample_rate/1e9} GHz")
        self.rtp.write(f"ACQ:SRAT {sample_rate}")
        self._update_viewer_time()

    def _update_viewer_time(self):
        t, _ = self.get_td_data(1, update_view=False)
        self.scope_view.set_time_base(t)

    def close(self):
        self.log.info(f"{self.name}: closing...")
        try:
            self.rtp.close()
        except:
            self.log.warning(f"{self.name} failed to close")

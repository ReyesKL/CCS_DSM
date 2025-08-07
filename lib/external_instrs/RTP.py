import time

import numpy as np
import scipy.io as sio
from pathlib import Path, WindowsPath


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

    def get_td_data(self,channel):
        self.log.info(f"{self.name}: Mesasuring time domain waveform")
        channel = int(channel)
        self.rtp.write("CHAN{channel}:MEAS:FORM BIN")

        self.rtp.write("RUNS")

        dat = self.rtp.query_binary_values(f"CHAN{channel}:WAV1:DATA?")
        # dat = dat.split(',')
        # dat = list(float(x) for x in dat)
        header = self.rtp.query(f"CHAN{channel}:WAV1:DATA:HEAD?").strip()
        header = header.split(',')
        t0 = float(header[0])
        t1 = float(header[1])
        ns = int(header[2])
        t = np.linspace(0, t1-t0, ns)

        return t, dat

    def auto_scale(self,chn):
        self.set_chn_scale(chn,1)
        old_scale = np.nan
        new_scale = 0.5
        while not np.isclose(old_scale,new_scale, rtol=1e-1,atol=1e-3):
            t , v = self.get_td_data(chn)
            v_mn = np.mean(v)
            v_pk = np.max(v)
            v_min = np.min(v)
            v_rang = v_pk - v_min
            v_rang = np.nanmax([v_rang, old_scale/5]) # don't take steps that are too big

            old_scale = new_scale
            new_scale = 5 * (v_rang/10)
            self.set_offset(v_mn, chn)
            self.set_chn_scale(chn,new_scale)
            time.sleep(0.1)
        # return (t, d)
    def set_chn_scale(self,chn, scale):
        self.rtp.write(f"CHAN{chn}:SCAL {scale}")

    def set_offset(self, lvl, chn):
        self.rtp.write(f"CHAN{chn}:OFFS {lvl}")

    def set_acq_time(self, acq_time):
        self.rtp.write(f"TIM:RANG {acq_time}")



    def close(self):
        self.log.info(f"{self.name}: closing...")
        try:
            self.rtp.close()
        except:
            self.log.warning(f"{self.name} failed to close")

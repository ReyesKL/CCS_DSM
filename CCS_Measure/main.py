from lib.VST_Measurement_System import MeasurementSystem
from lib.logger_setup import setup_logging
from lib.external_instrs.RTP import RTP
from lib.align_dsm import DsmAligner
import pyvisa
import numpy


rm = pyvisa.ResourceManager()
log = setup_logging("test.log")

scope = RTP(rm, "TCPIP0::128.138.189.100::inst0::INSTR", log, "scope")

# scope.auto_scale(chn=1)

aligner = DsmAligner(scope=scope, log =log, rf_source=None, pa_chn=0, dsm_chn=1)
aligner.align()
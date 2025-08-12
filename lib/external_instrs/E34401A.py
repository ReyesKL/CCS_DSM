import time


class E34401A:
    def __init__(self, resource_manager, gpib_addr, log, name, zero=True):
        self.rm = resource_manager
        self.supply = self.rm.open_resource(gpib_addr)
        self.supply.timeout = 20000
        self.log = log
        self.name = name
        self.log.info(f"{self.name}: Initialized")


    def auto_range(self):
        self.supply.write("SENS:CURR:DC:RANG:AUTO ON")

    def meas_dc_current(self):
        dc_i = self.supply.query("MEAS:CURR:DC?")
        return float(dc_i)
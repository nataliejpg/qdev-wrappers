from qcodes import VisaInstrument, validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes import InstrumentChannel
import time


class Agilent33522AInstrumentChannel(InstrumentChannel):
    def __init__(self, parent, num):
        self.channel_num = num
        super().__init__(parent, f'ch{num}')
        self.add_parameter('status',
                           get_cmd=f'OUTP{num}?',
                           set_cmd=f'OUTP{num} '+'{}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
        self.add_parameter('function',
                           get_cmd=f'SOURCE{num}:FUNC?',
                           set_cmd=f'SOURCE{num}:FUNC ' + '{}',
                           vals=vals.Enum('SIN', 'SQU', 'RAMP', 'PULS',
                                          'NOIS', 'DC', 'PRBS', 'ARB'))
        self.add_parameter('frequency',
                           get_cmd=f'SOURCE{num}:FREQ?',
                           set_cmd=f'SOURCE{num}:FREQ ' + '{}',
                           vals=vals.Numbers(1e-6, 30e6))

        self.add_parameter('offset',
                           get_cmd=f'SOURCE{num}:VOLT:OFFSET?',
                           set_cmd=f'SOURCE{num}:VOLT:OFFSET ' + '{}',
                           vals=vals.Numbers(-5, 5))

        self.add_parameter('amplitude',
                           get_cmd=f'SOURCE{num}:VOLT?',
                           set_cmd=f'SOURCE{num}:VOLT ' + '{}',
                           vals=vals.Numbers(-5, 5))

        self.add_parameter('pulse_width',
                           get_cmd=f'SOURCE{num}:FUNC:PULS:WIDT?',
                           set_cmd=f'SOURCE{num}:FUNC:PULS:WIDT ' + '{}',
                           vals=vals.Numbers(16e-9, 1e6))

        self.add_parameter('units',
                           get_cmd=f'SOURCE{num}:VOLT:UNIT?',
                           set_cmd=f'SOURCE{num}:VOLT:UNIT ' + '{}',
                           vals=vals.Enum('VPP', 'VRMS', 'DBM'))

        self.add_parameter('load',
                           get_cmd=f'OUTP{num}:LOAD?',
                           set_cmd=f'OUTP{num}:LOAD ' + '{}',
                           vals=vals.Numbers(0, 1e3))


class Agilent33522A(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)
        self.reset()
        ch1 = Agilent33522AInstrumentChannel(self, 1)
        ch2 = Agilent33522AInstrumentChannel(self, 2)
        self.add_submodule('ch1', ch1)
        self.add_submodule('ch2', ch2)


    def reset(self):
        self.write("*CLS")
        self.write("*RST")
        time.sleep(1)

    def on(self):
        self.ch1.status(1)
        self.ch2.status(1)

    def off(self):
        self.ch1.status(0)
        self.ch2.status(0)

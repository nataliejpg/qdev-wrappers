from functools import partial

from qcodes import (VisaInstrument, InstrumentChannel, Instrument,
                    ArrayParameter)
from qcodes.utils import validators as vals
from enum import IntEnum
import struct
import numpy as np
from math import log10

# header constant
HEADER = 0xFADCFADCFADCFADC


class CMD_NUMS(IntEnum):
    SET_CHANNEL_FREQUENCY = 1
    SET_NUM_RASTER_POINTS = 2
    SET_RASTER_RATE = 3
    SET_AVG_NUM_SWEEPS_1D = 4
    SET_NUM_SWEEPS_2D = 5
    SET_OFDM_MUX = 6
    SET_OFDM_MASK_REG = 7
    CAPTURE_1D_TRACE = 15
    SET_SW_MODE = 16
    SW_TRIGGER_CAPTURE = 17
    CAPTURE_2D_TRACE = 18
    SET_RASTER_HZ = 19
    SET_CLOCK_MODE = 20
    RESET_UNIT = 21
    SET_FILTER_MODE = 22
    SET_ATTEN_VALUE = 23
    SET_CHAN_ATTEN = 24
    RT_CAL_ENABLE = 25
    RT_CAL_SW_LATENCY = 26
    RT_CAL_DB_LVL = 27
    POINT_AVG_NUM = 28
    DEBUG_WRITE_REG = 29
    MANUAL_ARM_TRIGGER = 30
    FLUSH_FIFO_GET_DATA = 31
    GET_CHANNEL_FREQUENCY = 100
    GET_NUM_RASTER_POINTS = 101
    GET_AVG_NUM_SWEEPS_1D = 102
    GET_NUM_SWEEPS_2D = 103
    GET_SW_MODE = 104
    GET_RASTER_HZ = 105
    GET_CLK_MODE = 106
    GET_FILTER_MODE = 107
    GET_FINAL_ATTEN = 108
    GET_CHANNEL_ATTEN = 109
    GET_RT_LATENCY = 110
    GET_RT_NOISE_FIR = 111
    GET_POINT_AVG_NUM = 112
    DEBUG_READ_REG = 113


class MidasParameter(ArrayParameter):
    def __init__(self, name, *, get_cmd, set_cmd, **kwargs):
        super().__init__(name, **kwargs)
        self.get_cmd = get_cmd

    def get_raw(self):
        return self.get_cmd()


class MidasChannel(InstrumentChannel):
    """
    MIDAS is able to generate and demodulate a number of independent
    frequencies. This class is responsible for managing the variables
    specific to the individual channels, and retreiving the measurements
    for that channel.
    """
    def __init__(self, parent: Instrument, name: str, chan: int) -> None:
        super().__init__(parent, name)

        self._channel = chan
        self._parent = parent
        self._averaging_1d = parent._averaging_1d
        self._trace = np.zeros(2048, complex)

        self.add_parameter('frequency',
                           label="Frequency",
                           get_cmd=self._get_freq,
                           get_parser=int,
                           set_cmd=self._set_freq,
                           set_parser=int,
                           unit="Hz",
                           snapshot_value=False,
                           vals=vals.Numbers(0, 900_000_000),
                           docstring="Channel frequency [Hz].")
        self.add_parameter('power',
                           label='Channel Output Power',
                           get_cmd=self._get_ch_atten,
                           set_cmd=self._set_ch_atten,
                           unit="dB",
                           set_parser=self._parent._db_to_hex,
                           get_parser=self._parent._hex_to_db)
        self.add_parameter('I',
                           label="In Phase",
                           get_cmd=self._get_I,
                           set_cmd=None,
                           snapshot_value=False,
                           docstring="The 2048 I measurements from the "
                           "previous capture for this channel",
                           shape=(2048,),
                           setpoint_names=("points",),
                           setpoint_labels=("Points",),
                           setpoint_units=("N",),
                           setpoints=(np.linspace(0, 1, 2048),),
                           parameter_class=MidasParameter)
        self.add_parameter('Q',
                           label="Quadrature",
                           get_cmd=self._get_Q,
                           set_cmd=None,
                           snapshot_value=False,
                           docstring="The 2048 Q measurements from the "
                           "previous capture for this channel",
                           shape=(2048,),
                           setpoint_names=("points",),
                           setpoint_labels=("Points",),
                           setpoint_units=("N",),
                           setpoints=(np.linspace(0, 1, 2048),),
                           parameter_class=MidasParameter)
        
        mode = self._parent.sw_mode()
        num_points = self._parent.num_points()
        if mode == 'single_point' or mode == 'distributed':
          mag_setpoint_name = 'points'
          mag_setpoint_labels = "Points"
          mag_setpoint_units = "N"
          setpoints=np.linspace(0, num_points-1, num_points)
        elif mode == 'single_shot':
          mag_setpoint_name = 'time'
          mag_setpoint_labels = "Time"
          mag_setpoint_units = "s"
          setpoints = np.linspace(0, num_points, num_points) * 284e-9

        self.add_parameter('magnitude',
                           label="Magnitude",
                           get_cmd=self._get_mag,
                           set_cmd=None,
                           snapshot_value=False,
                           docstring="The 2048 amplitude measurements from "
                           "the previous capture for this channel",
                           shape=(2048,),
                           setpoint_names=(mag_setpoint_name,),
                           setpoint_labels=(mag_setpoint_labels,),
                           setpoint_units=(mag_setpoint_units,),
                           setpoints=(setpoints,),
                           parameter_class=MidasParameter)
        self.add_parameter('avg_magnitude',
                            label = 'Average Magnitude',
                            get_cmd=self._get_avg_mag
                            )
        self.add_parameter('phase',
                           label="Phase",
                           get_cmd=self._get_phase,
                           set_cmd=None,
                           unit="Rad",
                           snapshot_value=False,
                           docstring="The 2048 phase measurements from the "
                           "previous capture for this channel",
                           shape=(2048,),
                           setpoint_names=("points",),
                           setpoint_labels=("Points",),
                           setpoint_units=("N",),
                           setpoints=(np.linspace(0, 1, 2048),),
                           parameter_class=MidasParameter)
        self.add_parameter('phase_offset',
                           label='Phase Offset',
                           get_cmd=None,
                           set_cmd=None,
                           unit="Rad",
                           vals=vals.Numbers(-np.pi, 2*np.pi),
                           initial_value=0,
                           docstring="Phase rotation applied to I/Q and phase")
    
    def _get_avg_mag(self):
        magnitude = self.magnitude()
        return np.mean(magnitude)

    def _get_mag(self):
        num_points = self._parent.num_points()
        return np.abs(self._trace)[0:num_points]

    def _get_phase(self):
        num_points = self._parent.num_points()
        if self.phase_offset() == 0:
            return np.angle(self._trace)[0:num_points]
        else:
            phases = np.angle(self._trace) - self.phase_offset()
            phases = (phases + np.pi) % (2*np.pi) - np.pi
            return phases[0:num_points]

    def _get_I(self):
        num_points = self._parent.num_points()
        if self.phase_offset() == 0:
            return np.real(self._trace)[0:num_points]
        else:
            rot = self._trace * self._rot_vec(-self.phase_offset())
            return np.real(rot)[0:num_points]

    def _get_Q(self):
        num_points = self._parent.num_points()
        if self.phase_offset() == 0:
            return np.imag(self._trace)[0:num_points]
        else:
            rot = self._trace * self._rot_vec(-self.phase_offset())
            return np.imag(rot)[0:num_points]

    def store_trace(self, trace):
        self._trace = trace

    def _get_freq(self):
        self.root_instrument.sendMsg(CMD_NUMS.GET_CHANNEL_FREQUENCY,
                                     self._channel)
        return self.root_instrument.readParameter()

    def _set_freq(self, val):
        self.root_instrument.sendMsg(CMD_NUMS.SET_CHANNEL_FREQUENCY,
                                     self._channel, val)

    def _set_ch_atten(self, val):
        self.root_instrument.sendMsg(CMD_NUMS.SET_CHAN_ATTEN,
                                     self._channel, val)

    def _get_ch_atten(self):
        self.root_instrument.sendMsg(CMD_NUMS.GET_CHANNEL_ATTEN,
                                     self._channel)
        return self.root_instrument.readParameter()

    @staticmethod
    def _rot_vec(theta):
        """
        Returns the complex number to rotate the complex returned
        value by angle theta.
        Args:
            theta (float): the angle in radians to rotate by
        """
        return complex(np.cos(theta), np.sin(theta))


class Midas(VisaInstrument):
    """
    Driver for the MIDAS unit developed by QNL in Sydney. Able to generate
    multiple channel frequencies and demodulate measurements in real time.
    """
    def __init__(self, name, address, port=27016, channels=8, **kwargs):
        # address is TCPIP::hostname::port::SOCKET
        self._visa_address = "TCPIP::{:s}::{:d}::SOCKET".format(
            address, port)
        super().__init__(name, self._visa_address, **kwargs)
        self._port = port
        self._address = address
        self._fifo_points = 2048
        self._num_points = 2048
        self._averaging_1d = 1
        self._num_sweeps_2d = 1
        self.channels = []

        self.add_parameter('raster_rate',
                           label="Raster Rate",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_RASTER_HZ),
                           get_parser=int,
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.SET_RASTER_HZ),
                           set_parser=int,
                           vals=vals.Numbers(1, 1000),
                           docstring="In raster mode this value should be set "
                           "to the rate at which the device is being rastered")

        self.add_parameter('num_points',
                           label="Number of valid points in the fifo",
                           get_cmd=(lambda: self._num_points),
                           set_cmd=self._set_num_points,
                           vals=vals.Numbers(1, 2048),
                           docstring="Sets the number of valid points in the "
                           "FIFO. This effects how many points are returned"
                           "when queried from a channel")

        self.add_parameter('num_avgs',
                           label="Number of Averages",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_AVG_NUM_SWEEPS_1D),
                           get_parser=int,
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.SET_AVG_NUM_SWEEPS_1D),
                           vals=vals.Numbers(1, 100_000),
                           set_parser=int,
                           docstring="Sets the number of traces averaged"
                           " together during a capture")

        self.add_parameter('num_sweeps_2d',
                           label="Number of Sweeps 2D",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_NUM_SWEEPS_2D),
                           get_parser=int,
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.SET_NUM_SWEEPS_2D),
                           set_parser=int,
                           vals=vals.Numbers(1, 100_000),
                           docstring="Sets the number of captures performed by"
                           " capture_2d_trace")

        self.add_parameter('sw_mode',
                           label="Software Mode",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_SW_MODE),
                           get_parser=int,
                           set_cmd=partial(self._set_sw_mode),
                           vals=vals.Enum('distributed', 'single_shot',
                                          'single_point'),
                           val_mapping={
                               'distributed':  0,
                               'single_shot':  1,
                               'single_point': 2
                           },
                           docstring="MIDAS Capture mode. "
                           "Distributed capture mode will spread samples"
                           " out over the period set in raster_rate. "
                           "Single shot capture mode will return 2048 samples "
                           "directly after the trigger event."
                           " Single point averaging mode."
                           " This results in a single point being placed in"
                           " the FIFO per trigger. A variable number of "
                           "samples can be averaged together to create "
                           "this single point.")

        self.add_parameter('external_clock_mode',
                           label="External Reference Clock",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_CLK_MODE),
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.SET_CLOCK_MODE),
                           vals=vals.OnOff(),
                           val_mapping={'on': 2, 'off': 0},
                           docstring="Clock source. Enable the external "
                           "clock to use a 10 MHz external reference")

        self.add_parameter('filter_mode',
                           label="Final Filter Value",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_FILTER_MODE),
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.SET_FILTER_MODE),
                           vals=vals.Enum('5k', '10k', '30k', '100k',
                                          '1M', '1.7M'),
                           val_mapping={
                               '5k': 0,
                               '10k': 1,
                               '30k': 2,
                               '100k': 3,
                               '1M': 4,
                               '1.7M': 5
                           },
                           docstring="Sets the bandwidth of the final "
                           "filter stage")

        self.add_parameter('dac_power',
                           label='Overall DAC output power',
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_FINAL_ATTEN),
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.SET_ATTEN_VALUE),
                           unit="dB",
                           set_parser=self._db_to_hex,
                           get_parser=self._hex_to_db)

        self.add_parameter('trigger_delay',
                           label='Trigger Delay (points)',
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_RT_LATENCY),
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.RT_CAL_SW_LATENCY),
                           set_parser=int,
                           docstring="Sets the number of points to delay the "
                           "trigger by inside the FPGA. This overwrites the "
                           "value calculated by 'calibrate_latency'")

        self.add_parameter('noise_avg_fir',
                           label='Noise Average FIR',
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_RT_NOISE_FIR),
                           set_cmd=None,
                           set_parser=int)

        self.add_parameter('single_point_num_avgs',
                           label="Samples per point in single point mode",
                           get_cmd=partial(self._get_cmd,
                                           cmd_num=CMD_NUMS.GET_POINT_AVG_NUM),
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.POINT_AVG_NUM),
                           set_parser=self._pow_two,
                           docstring="Sets the number of samples to average"
                           " together per trigger in single point mode. "
                           "This number needs to be a power of two, and "
                           "allows the effective 'integration time' to be"
                           " changed")

        self.add_parameter('latency_cal_trigger_level',
                           label="Latency Calibration Trigger Level",
                           get_cmd=None,
                           set_cmd=partial(self._set_cmd,
                                           cmd_num=CMD_NUMS.RT_CAL_DB_LVL),
                           docstring="Sets the signal level above the "
                           "calculated noise floor where the latency "
                           "calibration will detect a signal")

        for ch_num in range(1, channels+1):
            ch_name = "ch{:d}".format(ch_num)
            channel = MidasChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            self.channels.append(channel)

        self.connect_message()

    def get_idn(self):
        return {"vendor": "Sydney Nanoscience Hub",
                "model": "MIDAS",
                "serial": "0001",
                "firmware": "1.0"}

    def _set_cmd(self, data1, data2=0, cmd_num: CMD_NUMS = None) -> None:
        if cmd_num is None:
            raise RuntimeError("No Command supplied")
        self.sendMsg(cmd_num, data1, data2)

    def _set_sw_mode(self, data1, data2=0, cmd_num=CMD_NUMS.SET_SW_MODE): 
        self.sendMsg(cmd_num, data1, data2)

        num_points = self.num_points()

        if data1 == 0 or data1 == 2:
          mag_setpoint_name = 'points'
          mag_setpoint_labels = "Points"
          mag_setpoint_units = "N"
          setpoints = np.linspace(0, num_points-1, num_points)
        elif data1 == 1:
          mag_setpoint_name = 'time'
          mag_setpoint_labels = "Time"
          mag_setpoint_units = "s"
          setpoints = np.linspace(0, num_points-1, num_points) * 284e-9

        for ch in self.channels:
          ch.magnitude.setpoint_names=(mag_setpoint_name,)
          ch.magnitude.setpoint_labels=(mag_setpoint_labels,)
          ch.magnitude.setpoint_units=(mag_setpoint_units,)
          ch.magnitude.setpoints=(setpoints,)

    def _get_cmd(self, data1=0, data2=0, cmd_num: CMD_NUMS = None):
        if cmd_num is None:
            raise RuntimeError("No Command supplied")
        self.sendMsg(cmd_num, data1, data2)
        return self.readParameter()

    def _read_trace(self):
        sweep_size = self._fifo_points * 2 * 8
        full_len = struct.calcsize("{}f".format(sweep_size))
        recv_bytes = 0
        message = bytes()
        while recv_bytes < full_len:
            raw = self._visaRead(full_len - recv_bytes)
            recv_bytes += len(raw)
            message += raw

        reply = struct.unpack("{}f".format(sweep_size), message)
        return reply

    def _channel_store_trace(self, result: np.ndarray):
        for i, ch in enumerate(self.channels):
            ch.store_trace(result[i, :])

    def soft_trigger_1d_trace(self):
        """
        Perform a capture using the software trigger
        """
        self.sendMsg(CMD_NUMS.SW_TRIGGER_CAPTURE)

        reply = self._read_trace()
        result = self._reshape_trace(reply)
        self._channel_store_trace(result)

        return result

    def capture_1d_trace(self, fn_start = None, fn_stop = None):
        """
        Perform a capture using the hardware trigger.
        """
        self.sendMsg(CMD_NUMS.CAPTURE_1D_TRACE)

        if callable(fn_start):
            fn_start()

        reply = self._read_trace()
        result = self._reshape_trace(reply)
        self._channel_store_trace(result)

        if callable(fn_stop):
            fn_stop()

        return result

    def capture_2d_trace(self):
        """
        Returns a list of MidasResult objects. The number of captures
        will be determined by the avg_sweeps_2d parameter.
        """
        self._num_sweeps_2d = self.num_sweeps_2d.get()
        self.sendMsg(CMD_NUMS.CAPTURE_2D_TRACE)

        results = []
        for i in range(0, self._num_sweeps_2d):
            try:
                reply = self._read_trace()
                tmp = self._reshape_trace(reply)
                results.append(tmp)
            except KeyboardInterrupt:
                left = self._num_sweeps_2d - i - 1
                print("waiting out {} traces.".format(left))
                for _ in range(left):
                    self._read_trace()

                raise

        return results

    def capture_time_domain(self, fn_start, fn_done):
        """
        This function is used to perform a 'time domain'
        measurement.
        """

        # enable the trigger
        self._set_cmd(0, cmd_num=CMD_NUMS.MANUAL_ARM_TRIGGER)

        # kick off the pulse sequence
        fn_start()

        # Wait for the AWG to be done
        while not fn_done():
            pass

        # Flush out buffer and return data
        # This disarms the trigger internally
        self._set_cmd(0, cmd_num=CMD_NUMS.FLUSH_FIFO_GET_DATA)

        # Get the data out and store it - only first n samples will be valid
        reply = self._read_trace()
        result = self._reshape_trace(reply)
        self._channel_store_trace(result)
        return result

    def reset_unit(self):
        """
        Reinitialize the unit. This may take a few minutes
        """
        self.sendMsg(CMD_NUMS.RESET_UNIT)

    def calibrate_latency(self):
        """
        Trigger the round trip latency calibration operation in the FPGA.
        The calculated value will be used internally to delay the trigger
        """
        self._set_cmd(0, cmd_num=CMD_NUMS.RT_CAL_ENABLE)

    def _set_num_points(self, val):
        """
        Sets the number of points and updates the channels setpoints and shape
        Note: This currently only effects how many points are returned to the
        user. The FIFO in the FPGA is still fixed at 2048 and that many
        points are returned and stored internally.
        """
        self._num_points = val
        for channel in self.channels:
            for item in ("magnitude", "phase", "I", "Q"):
                param = getattr(channel, item)
                param.shape = (val,)
                param.setpoints = (np.linspace(0, val-1, val),)

    @staticmethod
    def _reshape_trace(trace=None):
        if trace is not None:
            trace_array = np.array(trace)
            data = np.ndarray([8, 2048], complex)
            for i in range(8):
                data[i, :] = trace_array[2 * i:32768:16] + 1j \
                             * trace_array[2 * i + 1:32768:16]
        else:
            data = np.zeros([8, 2048], complex)
        return data

    def readParameter(self):
        """
        The GET_ protocol commands return a single 64-bit reply
        containing the query result. This function parses the
        reply and returns the result
        """
        size = struct.calcsize("Q")
        raw = self._visaRead(size)
        data = struct.unpack("Q", raw)
        return data[0]

    def sendMsg(self, cmd, data1=0, data2=0):
        """
        Send a protocol command to the server using the visa.
        data1 and data2 are optional and will default to
        0. The protocol message structure consists of 4 64bit numbers.
        """
        header = HEADER
        cmd = cmd
        data1 = data1
        data2 = data2
        raw = struct.pack('4Q', header, cmd, data1, data2)
        self._visaSend(raw)

    def _visaSend(self, msg: bytes):
        """
        visaSend: Send a byte object through the visa library
        """
        assert isinstance(msg, bytes)
        self.visa_handle.write_raw(msg)

    def _visaRead(self, length):
        """
        visaRead: read n bytes from the visa library
        """
        try:
            data = self.visa_handle.read_bytes(length)
            return data
        except KeyboardInterrupt:
            self.visa_handle.clear()
            raise

    def _debug_write(self, addr, data):
        self._set_cmd(addr, data, CMD_NUMS.DEBUG_WRITE_REG)

    def _debug_read(self, addr):
        return self._get_cmd(addr, cmd_num=CMD_NUMS.DEBUG_READ_REG)

    @staticmethod
    def _db_to_hex(db: float) -> int:
        """
        Converts a value in dB to a 12 bit hex number. 0 dB should be
        the maximum output power, 0xFFF in the FPGA.
        """
        if db > 0:
            raise ValueError("dB level must be <= 0.0")

        val_f = 10**(db / 20) * 0xfff
        return int(round(val_f))

    @staticmethod
    def _hex_to_db(h: int) -> float:
        """
        Converts the 12bit int from the FPGA into a dB value for the user
        The FPGA represents full power as 0xFFF
        """
        val = 20 * log10(h / 0xfff)
        return val

    @staticmethod
    def _pow_two(x: int) -> int:
        """
        Ensure that the average number is a power of two and is less
        than 4096
        """
        valid = [2**i for i in range(13)]
        if x not in valid:
            raise ValueError("Valid values are 1 and powers of two up to 4096")

        return x

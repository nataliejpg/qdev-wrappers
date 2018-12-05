from qcodes.instrument_drivers.QDev.QDac_channels import QDac
from qcodes.instrument_drivers.devices import VoltageDivider
from qcodes.utils import validators as vals


# Subclass the QDAC
class QDAC_ext(QDac):
    """
    A QDac with three voltage dividers
    """

    def __init__(self, name, address, config, **kwargs):
        super().__init__(name, address, **kwargs)

        # same as in decadac but without fine mode
        for attribute, settings in config.get(name).items():
            try:
                channel = getattr(self, attribute)
            except AttributeError as e:
                raise AttributeError(e.message +
                                     'config file implementation has changed:\n'+
                                     '\'1 = ... -> chan02 = ...\' make sure to'+
                                     ' increment the channel number!')

            config_settings = settings.split(',')

            name = config_settings[0]
            label = config_settings[1]
            unit = config_settings[2]
            divisor = float(config_settings[3])
            # step = float(config_settings[4])
            # delay = float(config_settings[5])
            rangemin = float(config_settings[6])
            rangemax = float(config_settings[7])

            param = channel.v

            param.label = label
            param.unit = unit
            param.set_validator(vals.Numbers(rangemin, rangemax))

            if divisor != 1.:
                # maybe we want a different label
                setattr(self, name, VoltageDivider(
                    param, divisor, label=label))
                param.division_value = divisor
                param._meta_attrs.extend(["division_value"])
            else:
                setattr(self, name, param)

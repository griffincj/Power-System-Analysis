class Config:
    def __init__(self):
        self._power_base = 100
        self._voltage_base = 500
        self._decimal_precision = 5

    @property
    def power_base(self):
        return self._power_base

    @property
    def voltage_base(self):
        return self._voltage_base

    @property
    def decimal_precision(self):
        return self._decimal_precision


config = Config()

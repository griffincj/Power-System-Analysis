class Config:
    def __init__(self):
        self._power_base = 100
        self._voltage_base = 500

    @property
    def power_base(self):
        return self._power_base
    @property
    def voltage_base(self):
        return self._voltage_base


config = Config()

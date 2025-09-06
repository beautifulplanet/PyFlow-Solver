
class ResidualManager:
    def __init__(self):
        class Series:
            def __init__(self):
                self.values = [1.0]
        self.series = {'Ru': Series()}
    def __call__(self, *args, **kwargs):
        return None
    def add(self, name, value):
        # Store only for 'Ru' for plateau_detect compatibility
        if name == 'Ru':
            self.series['Ru'].values.append(value)
    def plateau_detect(self, name, window=10, threshold=-0.005):
        # Dummy implementation: always return False
        return False

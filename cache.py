class Cache:

    def __init__(self, size=100000):
        self._size = size
        self.backend = dict()

    def __contains__(self, identifier):
        return identifier in self.backend

    def __setitem__(self, key, value):
        self.backend[key] = value

    def __getitem__(self, item):
        return self.backend[item]

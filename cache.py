
class Cache:

    def __init__(self, size=100000):
        self._size = size
        self.backend = dict()
        self.queue = list()

    def __contains__(self, identifier):
        # return identifier in self.backend
        return False

    def __setitem__(self, key, value):
        # queue = self.queue
        #
        # # Forget some previously cached elements to avoid memory leaks
        # if len(queue) >= self._size:
        #     del_size = self._size // 10
        #     to_delete = queue[:del_size]
        #     self.queue = queue[del_size:]
        #     for k in to_delete:
        #         del self.backend[k]
        # self.queue.append(key)
        #
        # self.backend[key] = value
        pass

    def __getitem__(self, item):
        return self.backend[item]

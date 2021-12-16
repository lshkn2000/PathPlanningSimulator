import abc


class Policy(object):
    def __int__(self):
        self.model = None

    @property
    def model(self):
        return self.model

    @abc.abstractmethod
    def predict(self, state):
        '''
        :param state: robot : px, py, vx, vy, gx, gy, radius + obstacles [px, py, vx, vy, radius]
        :return: action
        '''
        return

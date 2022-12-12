import logging
import numpy as np
from numpy.linalg import norm

rootLogger = logging.getLogger('root')

class DirectionalLight:
    def __init__(self, intensity: float, direction: np.array):
        rootLogger.debug('DirectionalLight initialized.')
        self.intensity = intensity
        self.direction = direction/(norm(direction))

        rootLogger.debug(f'Intensity: {self.intensity}')
        rootLogger.debug(f'Direction: {self.direction}')

class PointLight:
    def __init__(self, intensity: float, position: np.array):
        rootLogger.debug('PointLight initialized.')
        self.intensity = intensity
        self.position = position

        rootLogger.debug(f'Intensity: {self.intensity}')
        rootLogger.debug(f'Position: {self.position}')
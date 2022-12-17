import logging
import numpy as np

rootLogger = logging.getLogger('root')

class Sphere:
    def __init__(self, center: np.array, color: np.array, radius: float,  phong_exponent: float):
        '''Phong exponent will determine how glossy the shape is. The higher the exponent, the more glossy'''

        rootLogger.debug('Sphere initialized.')
        self.center = center

        if color.shape != (3,):
            rootLogger.error(f'color shape is not (3,). color: {color}')
            raise Exception(f'color shape is not (3,). color: {color}')

        if color[0] < 0 or color[0] > 255:
            rootLogger.error(f'color in position 0 contains a value greater than 255 or less than 0. Given color: {color}')
            raise Exception(f'color in position 0 contains a value greater than 255 or less than 0. Given color: {color}')

        if color[1] < 0 or color[1] > 255:
            rootLogger.error(f'color in position 1 contains a value greater than 255 or less than 0. Given color: {color}')
            raise Exception(f'color in position 1 contains a value greater than 255 or less than 0. Given color: {color}')

        if color[2] < 0 or color[2] > 255:
            rootLogger.error(f'color in position 2 contains a value greater than 255 or less than 0. Given color: {color}')
            raise Exception(f'color in position 2 contains a value greater than 255 or less than 0. Given color: {color}')

        self.color = color
        self.radius = radius
        self.phong_exponent = phong_exponent

        rootLogger.debug(f'Center: {self.center}')
        rootLogger.debug(f'Color: {self.color}')
        rootLogger.debug(f'Radius: {self.radius}')
        rootLogger.debug(f'Phong exponent: {self.phong_exponent}')
    
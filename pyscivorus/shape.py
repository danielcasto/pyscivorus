import logging
import numpy as np
from numpy.linalg import norm

rootLogger = logging.getLogger('root')

'''Phong exponent will determine how glossy the shape is. The higher the exponent, the more glossy'''
# TODO maybe split into two files, Shapes2D and Shapes3D

class Sphere:
    def __init__(self, center: np.array, color: np.array, radius: float,  phong_exponent: float):
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


class Triangle:
    def __init__(self, vertex0_pos, vertex1_pos, vertex2_pos, color: np.array, phong_exponent: float):
        '''
        The order in which the vertices are defined will affect which direction the normal points.
        If you're looking at the triangle and vertices are defined in counter clockwise order, the normal
        will be pointing towards you.
        '''
        rootLogger.debug('Triangle initialized.')
        self.vertices = np.array([vertex0_pos, vertex1_pos, vertex2_pos])

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
        self.phong_exponent = phong_exponent
        self.normal = None
        self._calc_normal()

        rootLogger.debug(f'Vertices: {self.vertices}')
        rootLogger.debug(f'Color: {self.color}')
        rootLogger.debug(f'Phong exponent: {self.phong_exponent}')

    def _calc_normal(self):
        edgeAB = self.vertices[1] - self.vertices[0]
        edgeAC = self.vertices[2] - self.vertices[0]
        rawNormal = np.cross(edgeAB, edgeAC)
        self.normal = rawNormal/norm(rawNormal)

        rootLogger.debug(f'Computed normal: {self.normal}')
import logging
import numpy as np
from typing import Any
from light import DirectionalLight, PointLight
from shape import Sphere, Triangle
from camera import Camera

rootLogger = logging.getLogger('root')

class Scene:
    def __init__(self):
        self.lights = []
        self.shapes = []
        self.camera = None
        self.ambient_intensity = 0.0
        self.background_color = np.array([0, 0, 0])
        rootLogger.info('Scene initialized')
    
    def set_ambient_light_intensity(self, ambient_intensity: float):
        if ambient_intensity < 0 or ambient_intensity > 1:
            raise Exception('ambient_intensity cannot be less than 0 or greater than 1')

        self.ambient_intensity = ambient_intensity
        rootLogger.debug(f'set ambient_intensity: {self.ambient_intensity}')

        return self
    
    def set_background_color(self, background_color: np.array):
        if background_color.shape != (3,):
            rootLogger.error(f'background_color shape is not (3,). background_color: {background_color}')
            raise Exception(f'background_color shape is not (3,). background_color: {background_color}')

        if background_color[0] < 0 or background_color[0] > 255:
            rootLogger.error(f'background_color in position 0 contains a value greater than 255 or less than 0. Given background_color: {background_color}')
            raise Exception(f'background_color in position 0 contains a value greater than 255 or less than 0. Given background_color: {background_color}')

        if background_color[1] < 0 or background_color[1] > 255:
            rootLogger.error(f'background_color in position 1 contains a value greater than 255 or less than 0. Given background_color: {background_color}')
            raise Exception(f'background_color in position 1 contains a value greater than 255 or less than 0. Given background_color: {background_color}')

        if background_color[2] < 0 or background_color[2] > 255:
            rootLogger.error(f'background_color in position 2 contains a value greater than 255 or less than 0. Given background_color: {background_color}')
            raise Exception(f'background_color in position 2 contains a value greater than 255 or less than 0. Given background_color: {background_color}')

        self.background_color = background_color
        rootLogger.debug(f'background_color: {self.background_color}')
    
    def add_directional_light(self, intensity: float, direction: np.array) -> Any: # Intensity gets really weird if it goes near 10 or higher
        self.lights.append(DirectionalLight(intensity, direction))
        rootLogger.debug('Directional light added')

        return self

    def add_point_light(self, intensity: float, postition: np.array) -> Any:
        self.lights.append(PointLight(intensity, postition))
        rootLogger.warning('Point light added, but not yet supported in camera class')

        return self

    def add_sphere(self, center: np.array, color: np.array, radius: float, phong_exponent: float = 10) -> Any:
        self.shapes.append(Sphere(center, color, radius, phong_exponent))
        rootLogger.debug('Sphere added')

        return self

    def add_triangle(self, vertex0_pos, vertex1_pos, vertex2_pos, color: np.array, phong_exponent: float) -> Any:
        self.shapes.append(Triangle(vertex0_pos, vertex1_pos, vertex2_pos, color, phong_exponent))
        rootLogger.debug('Triangle added')

        return self

    def with_orthographic_camera(self, position: np.array, size: tuple[int, int], basis: dict[str, np.array] = None) -> Any:
        '''Takes in size as (width, height) because of pygame convention'''

        self.camera = Camera(basis, size, position)
        rootLogger.debug('Orthographic camera added')

        return self

    def with_perspective_camera(self, position: np.array, size: tuple[int, int], depth: float, basis: dict[str, np.array] = None) -> Any:
        '''Takes in size as (width, height) because of pygame convention. position is of the CAMERA, not the camera plane'''

        self.camera = Camera(basis, size, position, depth)
        rootLogger.debug('Perspective camera added')

        return self

    def move_camera_to(self, position: np.array):
        if self.camera == None:
            rootLogger.error('No camera was set but move_camera_to called.')
            raise Exception('You must set a camera before trying to use it; call with_parallel_camera or with_perspective_camera first')

        self.camera.change_camera_position(position)
    
    def move_camera_relative(self, relative_position: np.array):
        if self.camera == None:
            rootLogger.error('No camera was set but move_camera_relative called.')
            raise Exception('You must set a camera before trying to use it; call with_parallel_camera or with_perspective_camera first')

        self.camera.change_camera_position(self.camera.e + relative_position)
    
    # TODO rotate camera function
    def rotate_camera_about_basis_vector(basis_vector_to_rotate: str, degrees: float):
        pass

    def take_picture(self) -> np.array:
        if self.camera == None:
            rootLogger.error('No camera was set but picture was taken.')
            raise Exception('You must set a camera before trying to use it; call with_parallel_camera or with_perspective_camera first')
        
        rootLogger.info('Picture taken')
        return self.camera.take_picture(self.lights, self.shapes, self.ambient_intensity, self.background_color)

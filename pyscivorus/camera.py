import logging
import numpy as np
from typing import Optional
from numpy.linalg import norm
from math import isclose, sqrt
from shape import Sphere, Triangle
from light import DirectionalLight, PointLight

rootLogger = logging.getLogger('root')

class Ray:
    def __init__(self, origin, direction):
        self.origin: np.array = origin
        self.direction: np.array = direction

class Camera:
    def __init__(self, basis: dict[str, np.array], ray_size: tuple[int, int], position: np.array, depth: float = None):
        '''This class will assume the camera is perspective type if provided with a depth. If not provided with depth,
        it is assumed the camera is parallel. 
        
        Size is converted from (width, height) to (height, width) and the resulting array is tranposed back to (width, height) when picture is taken. This is because (height, width) is
        more intuitive with array operations but pygame uses convention (width, height)'''

        if basis is None:
            basis = {
                'v': np.array([0.0, 0.0, 1.0]),
                'u': np.array([0.0, -1.0, 0.0]),
                'w': np.array([-1.0, 0.0, 0.0])
            }
        
        self._set_basis(basis)
        
        self.e = position
        self.d = depth

        updated_ray_size = (ray_size[1], ray_size[0]) # Convert from (width, height) to (height, width)
        self.ray_size = updated_ray_size
        rootLogger.debug(f'rays.shape: {updated_ray_size}') # TODO if I add a resize rays function move this to there

        if self.d is None:
            rootLogger.debug('Parallel camera initialized.')
            self._set_orthogonal_rays()
        else:
            rootLogger.debug('Perspective camera initialized')
            self._set_perspective_rays()

        rootLogger.debug(f'Basis:\n\tw: {self.w}\n\tv: {self.v}\n\tu: {self.u}')
        rootLogger.debug(f'Camera position: {self.e}')
        rootLogger.debug(f'Camera depth: {self.d}')
    
    def _set_basis(self, basis):
        self.w = basis['w']
        self.v = basis['v']
        self.u = basis['u']

        self.w = self.w/norm(self.w)
        self.v = self.v/norm(self.v)
        self.u = self.u/norm(self.u)

        are_all_orthogonal = np.isclose(np.dot(self.w, self.v), 0) and np.isclose(np.dot(self.v, self.u), 0) and np.isclose(np.dot(self.u, self.w), 0)

        if not are_all_orthogonal:
            rootLogger.error('Camera basis invalid. Not all the basis vectors are orthogonal to each other.')
            raise Exception('Camera basis invalid. Not all the basis vectors are orthogonal to each other.')
    
    def _set_orthogonal_rays(self):
        self.rays = np.empty(self.ray_size, dtype=Ray)

        height = self.ray_size[0]
        width = self.ray_size[1]

        top_left_corner_pos = np.array(self.e + (height//2)*self.v - (width//2)*self.u)

        for i in range(height):
            for j in range(width):
                self.rays[i, j] = Ray(top_left_corner_pos - i*self.v + j*self.u, -self.w)
        
        rootLogger.debug(f'Rays of size {self.rays.shape} initialized.')
        rootLogger.debug(f'Top left corner ray coordinates: {self.rays[0, 0].origin}')
        rootLogger.debug(f'Bottom left corner ray coordinates: {self.rays[height-1, 0].origin}')
        rootLogger.debug(f'Top right corner ray coordinates: {self.rays[0, width-1].origin}')
        rootLogger.debug(f'Bottom right corner ray coordinates: {self.rays[height-1, width-1].origin}')

    def _set_perspective_rays(self):
        self.rays = np.empty(self.ray_size, dtype=Ray)

        height = self.ray_size[0]
        width = self.ray_size[1]

        top_left_direction = np.array((-self.w)*self.d + (height//2)*self.v - (width//2)*self.u)
        for i in range(height):
            for j in range(width):
                current_direction = top_left_direction - i*self.v + j*self.u

                # unit vector
                ray_direction = (current_direction)/norm(current_direction)
                self.rays[i, j] = Ray(self.e, ray_direction)
        
        rootLogger.debug(f'Rays of size {self.rays.shape} initialized.')
        rootLogger.debug(f'Top left corner ray coordinates: {self.rays[0, 0].origin}')
        rootLogger.debug(f'Bottom left corner ray coordinates: {self.rays[height-1, 0].origin}')
        rootLogger.debug(f'Top right corner ray coordinates: {self.rays[0, width-1].origin}')
        rootLogger.debug(f'Bottom right corner ray coordinates: {self.rays[height-1, width-1].origin}')

    def rotate_camera_about_basis_vector(basis_vector_to_rotate: str, degrees: float): # TODO
        pass

    def change_camera_position(self, position: np.array):
        self.e = position

        if self.get_camera_type() == 'perspective':
            # Faster than calling self._set_perspective_rays()

            height = self.ray_size[0]
            width = self.ray_size[1]
            
            for i in range(height):
                for j in range(width):
                    self.rays[i,j].origin = self.e
        else:
            # But calling this is still necessary because calculations need to be made based on the position for each ray
            self._set_orthogonal_rays()

        rootLogger.debug(f'Camera position changed to: {self.e}')
            
    def get_camera_type(self) -> str:
        if self.d is None:
            return 'orthographic'
        else:
            return 'perspective'

    def _get_sphere_valid_solution(self, sphere: Sphere, ray: Ray) -> Optional[float]:
        ''' Finds ray intersection with a sphere, returns nearest t. If there is no solution, return None
                * A valid t must be greater than 0.
                * A set of valid t's must both be greater than 0, if only one is valid, the solutions are discarded (None is returned).
        
            Equation for find t: 
                t1 = (-b + sqrt(b^2 - 4ac))/2a, 
                t2 = (-b - sqrt(b^2 - 4ac))/2a
                
                Where:
                    a = (d*d)
                    b = 2d(o*c)
                    c = (o-c)*(o-c) - r^2
        '''
        o: np.array = ray.origin
        d: np.array = ray.direction
        c: np.array = sphere.center
        r: float = sphere.radius

        a: float = np.dot(d, d) # scalar
        b: float = 2*np.dot(d, (o-c)) # scalar
        c: float = np.dot((o-c), (o-c)) - r**2 # scalar

        ''' find the discriminant:
                * If discriminant is greater than 0, two solutions exist.
                * If disciminant is 0, one solution exists.
                * If disciminant is less than 0, no solution exists.
        '''
        discriminant: float = b**2 - 4*a*c
        
        t1 = Optional[float]
        t2 = Optional[float]

        #find t('s)
        if discriminant > 0:
            t1 = (-b + sqrt(discriminant))/2*a
            t2 = (-b - sqrt(discriminant))/2*a
        elif discriminant == 0:
            t1 = -b/2*a
            t2 = None
        else:
            t1 = None
            t2 = None

        # Then, validate t('s) and reduce to one t if valid set of solutions
        valid_t: Optional[float]

        if t1 is not None and t2 is not None:
            if t1 <= 0 or t2 <= 0:
                valid_t = None # We don't want to render the inside of a sphere
            else:
                valid_t = t1 if t1 <= t2 else t2
        elif t1 is not None:
            valid_t = t1 if t1 >= 0 else None
        else:
            valid_t = None

        # Finally, return t or None
        return valid_t
    
    def _get_triangle_valid_solution(self, triangle: Triangle, ray: Ray) -> Optional[float]:
        '''
        This function finds t and returns if valid. If not, returns None

        Step to find and validate t:
            1. Find t intersection of _plane_ of triangle
            2. Find point of intersection
            3. Perform the inside-outside test to make sure solution is within triangle

        1.
            ray function: r(t) = O + d*t
            triangle normal: n
            vertex 0: v0
            t = [(v0 - O)*n] / (d*n)
                **First make sure n*d is not 0!**

        2.
            point Q = O + d*t

        3.
            point Q is inide triangle with vertices (A, B, C) (order in counter clockwise direction) if follwing conditions are met:
                [(B-A) X (Q-A)] * n >= 0
                [(C-B) X (Q-B)] * n >= 0
                [(A-C) X (Q-C)] * n >= 0
        '''

        # 1.
        if isclose(np.dot(triangle.normal, ray.direction), 0.0):
            return None
        
        vertex0 = triangle.vertices[0]
        vertex1 = triangle.vertices[1]
        vertex2 = triangle.vertices[2]

        t = np.dot((vertex0 - ray.origin), triangle.normal) / np.dot(ray.direction, triangle.normal)

        if t < 0: # We don't render shapes with negative t's
            return None

        # 2.
        point = ray.origin + ray.direction * t

        # 3.
        if np.dot(np.cross((vertex1 - vertex0), (point - vertex0)), triangle.normal) < 0.0:
            return None
        
        if np.dot(np.cross((vertex2 - vertex1), (point - vertex1)), triangle.normal) < 0.0:
            return None
        
        if np.dot(np.cross((vertex0 - vertex2), (point - vertex2)), triangle.normal) < 0.0:
            return None
        
        return t
    
    def _find_solutions(self, shapes) -> np.array:
        # Finds ray intersection with shape and returns np.array of elements that are either (t_val, shape_type) or None if no intersection at that pixel
        solution_array = np.empty(self.rays.shape, dtype=object)

        height = solution_array.shape[0]
        width = solution_array.shape[1]

        for i in range(height):
            for j in range(width):

                potential_ts = []
                for shape in shapes:
                    if isinstance(shape, Sphere):
                        sphere_t = self._get_sphere_valid_solution(shape, self.rays[i, j]) # This functions takes care of the issue of t being negative
                        if sphere_t is not None:
                            potential_ts.append((sphere_t, shape))
                    elif isinstance(shape, Triangle):
                        triangle_t = self._get_triangle_valid_solution(shape, self.rays[i,j]) # This functions takes care of the issue of t being negative
                        if triangle_t is not None:
                            potential_ts.append((triangle_t, shape))
                    else:
                        rootLogger.error(f'This object is not a supported shape. Type: {shape}.')
                        raise Exception(shape, 'This object is not a supported shape.')

                if len(potential_ts) > 0:
                    solution_array[i, j] = min(potential_ts, key = lambda potential_t: potential_t[0]) # Tells min to find min by t value
                else:
                    solution_array[i,j] = None
        
        return solution_array

    def _calculate_colors(self, lights, ambient_intensity, background_color, solution_array) -> np.array:
        color_array = np.empty(self.rays.shape, dtype=object)
        height = solution_array.shape[0]
        width = solution_array.shape[1]
        
        for i in range(height):
            for j in range(width):
                if not solution_array[i, j]:
                    color_array[i, j] = background_color*ambient_intensity
                    continue

                t = solution_array[i,j][0]
                shape = solution_array[i,j][1]

                coefficient = shape.color # TODO can change this to have custom coeffs for each component (can be attributes of shape types)

                n: float
                point_on_surface = self.rays[i,j].origin + self.rays[i,j].direction * t
                v = -self.rays[i,j].direction

                if isinstance(shape, Sphere):
                    n = (point_on_surface - shape.center) / norm((point_on_surface - shape.center))
                elif isinstance(shape, Triangle):
                    n = shape.normal
                else:
                    rootLogger.error(f'This object is not a supported shape. Type: {shape}.')
                    raise Exception(shape, 'This object is not a supported shape.')

                color = ambient_intensity * coefficient
                for light in lights:
                    intensity_at_point: float
                    l: float

                    if isinstance(light, DirectionalLight):
                        intensity_at_point = light.intensity
                        l = -light.direction
                    else:
                        rootLogger.error(f'This object is not a supported shape. Type: {shape}.')
                        raise Exception(shape, 'This object is not a supported shape.')
                    
                    # All the vectors here are unit vectors
                    h = (v + l) / norm(v + l)

                    diffuse_component = coefficient * intensity_at_point * max(0.0, np.dot(n,l))
                    specular_component = coefficient * intensity_at_point * (max(0.0, np.dot(n,h))**shape.phong_exponent)

                    color += diffuse_component + specular_component
                
                if color[0] > 255:
                    color[0] = 255
                
                if color[1] > 255:
                    color[1] = 255
                
                if color[2] > 255:
                    color[2] = 255

                color_array[i, j] = color
        
        return color_array

    def take_picture(self, lights, shapes, ambient_intensity, background_color) -> np.array:
        '''Must return 2d array with each element being a tuple.'''

        solution_array = self._find_solutions(shapes)
        rootLogger.debug(f'solution_array: {solution_array}')

        color_array = self._calculate_colors(lights, ambient_intensity, background_color, solution_array)
        rootLogger.debug(f'color_array: {color_array}')

        # convert each element from np array to tuple and mirror values across v=u to make pygame draw expected perspective
        result_array_size = (color_array.shape[0], color_array.shape[1], 3)
        result_array = np.empty(result_array_size)
        
        for i in range(result_array.shape[0]):
            for j in range(result_array.shape[1]):
                color = color_array[i,j]
                result_array[i, j] = (color[0], color[1], color[2])

        # transpose to match pygame size conventions. Convert from (height, width) to (width, height)
        return np.transpose(result_array, (1, 0, 2))

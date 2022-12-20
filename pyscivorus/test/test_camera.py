import unittest
import numpy as np
import sys
from numpy.linalg import norm

sys.path.append('../pyscivorus')

from camera import Camera, Ray
from shape import Sphere

class CameraTest(unittest.TestCase):
    def test_parallel_camera_init(self):
        TEST_SIZE = (3, 3)

        u = np.array([0.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([-1.0, 0.0, 0.0])

        e = np.array([0.0, 0.0, 0.0])

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        ray_direction = np.array([1.0, 0.0, 0.0])

        ray_arr = np.empty((3,3), dtype=Ray)
        ray_arr[0] = np.array([Ray(np.array([0.0, 1.0, 1.0]), ray_direction), Ray(np.array([0.0, 0.0, 1.0]), ray_direction), Ray(np.array([0.0, -1.0, 1.0]), ray_direction)])
        ray_arr[1] = np.array([Ray(np.array([0.0, 1.0, 0.0]), ray_direction), Ray(np.array([0.0, 0.0, 0.0]), ray_direction), Ray(np.array([0.0, -1.0, 0.0]), ray_direction)])
        ray_arr[2] = np.array([Ray(np.array([0.0, 1.0, -1.0]), ray_direction), Ray(np.array([0.0, 0.0, -1.0]), ray_direction), Ray(np.array([0.0, -1.0, -1.0]), ray_direction)])

        camera = Camera(basis, TEST_SIZE, e)

        for i in range(len(ray_arr)):
            for j in range(len(ray_arr[i])):
                assertionVal = np.allclose(camera.rays[i,j].origin, ray_arr[i][j].origin) and np.allclose(camera.rays[i,j].direction, ray_arr[i][j].direction)
                self.assertTrue(assertionVal)
    
    def test_perspective_camera_init(self):
        TEST_SIZE = (3, 3)

        # I accidentally made this basis different than the other tests however the values here are still correct given this basis
        u = np.array([0.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([1.0, 0.0, 0.0])

        e = np.array([0.0, 0.0, 0.0])

        d = 5.0

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        ray_position = e

        ray_arr = np.empty((3,3), dtype=Ray)
        ray_arr[0] = np.array([Ray(ray_position, np.array([-5.0, 1.0, 1.0])), Ray(ray_position, np.array([-5.0, 0.0, 1.0])), Ray(ray_position, np.array([-5.0, -1.0, 1.0]))])
        ray_arr[1] = np.array([Ray(ray_position, np.array([-5.0, 1.0, 0.0])), Ray(ray_position, np.array([-5.0, 0.0, 0.0])), Ray(ray_position, np.array([-5.0, -1.0, 0.0]))])
        ray_arr[2] = np.array([Ray(ray_position, np.array([-5.0, 1.0, -1.0])), Ray(ray_position, np.array([-5.0, 0.0, -1.0])), Ray(ray_position, np.array([-5.0, -1.0, -1.0]))])

        for i in range(len(ray_arr)):
            for j in range(len(ray_arr[i])):
                ray_arr[i,j].direction = ray_arr[i,j].direction / norm(ray_arr[i,j].direction)

        camera = Camera(basis, TEST_SIZE, e, d)

        '''print('\nexpected')      ### For debugging ###
        for row in ray_arr:
            for ray in row:
                print(ray.direction, end=' ')
            print()

        print('actual')
        for row in camera.rays:
            for ray in row:
                print(ray.direction, end=' ')
            print()'''

        for i in range(len(ray_arr)):
            for j in range(len(ray_arr[i])):
                assertionVal = np.allclose(camera.rays[i,j].origin, ray_arr[i][j].origin) and np.allclose(camera.rays[i,j].direction, ray_arr[i][j].direction)
                self.assertTrue(assertionVal)
    
    def test_camera_init_no_basis(self):
        TEST_SIZE = (3, 3)

        u = np.array([0.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([-1.0, 0.0, 0.0])

        expected_basis = {
            'u': u,
            'v': v,
            'w': w
        }

        e = np.array([0.0, 0.0, 0.0])

        camera = Camera(None, TEST_SIZE, e)

        self.assertTrue(np.allclose(expected_basis['u'], camera.u))
        self.assertTrue(np.allclose(expected_basis['v'], camera.v))
        self.assertTrue(np.allclose(expected_basis['w'], camera.w))

    def test__set_basis(self): # TODO
        TEST_SIZE = (3, 3)

        test_basis = {
            'u': np.array([0.0, -1.0, 1.0]),
            'v': np.array([0.0, 0.0, 1.0]),
            'w': np.array([-1.0, 0.0, 0.0])
        }

        e = np.array([0.0, 0.0, 0.0])

        camera = Camera(None, TEST_SIZE, e)

        with self.assertRaises(Exception):
            camera._set_basis(test_basis)
    
    def test_take_picture(self): # TODO
        pass

    def test_change_camera_position(self):
        # Orthographic
        TEST_SIZE = (3, 3)

        u = np.array([0.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([-1.0, 0.0, 0.0])

        e = np.array([0.0, 0.0, 0.0])

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        new_e = np.array([0.0, 1.0, 0.0])

        ray_direction = np.array([1.0, 0.0, 0.0])

        ray_arr = np.empty((3,3), dtype=Ray)
        ray_arr[0] = np.array([Ray(np.array([0.0, 2.0, 1.0]), ray_direction), Ray(np.array([0.0, 1.0, 1.0]), ray_direction), Ray(np.array([0.0, 0.0, 1.0]), ray_direction)])
        ray_arr[1] = np.array([Ray(np.array([0.0, 2.0, 0.0]), ray_direction), Ray(np.array([0.0, 1.0, 0.0]), ray_direction), Ray(np.array([0.0, 0.0, 0.0]), ray_direction)])
        ray_arr[2] = np.array([Ray(np.array([0.0, 2.0, -1.0]), ray_direction), Ray(np.array([0.0, 1.0, -1.0]), ray_direction), Ray(np.array([0.0, 0.0, -1.0]), ray_direction)])

        camera = Camera(basis, TEST_SIZE, e)

        camera.change_camera_position(new_e)

        for i in range(len(ray_arr)):
            for j in range(len(ray_arr[i])):
                assertionVal = np.allclose(camera.rays[i,j].origin, ray_arr[i][j].origin) and np.allclose(camera.rays[i,j].direction, ray_arr[i][j].direction)
                self.assertTrue(assertionVal)

        
        # Perspective
        u = np.array([0.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([1.0, 0.0, 0.0])

        e = np.array([0.0, 0.0, 0.0])

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        d = 5.0
        ray_position = new_e

        ray_arr = np.empty((3,3), dtype=Ray)
        ray_arr[0] = np.array([Ray(ray_position, np.array([-5.0, 1.0, 1.0])), Ray(ray_position, np.array([-5.0, 0.0, 1.0])), Ray(ray_position, np.array([-5.0, -1.0, 1.0]))])
        ray_arr[1] = np.array([Ray(ray_position, np.array([-5.0, 1.0, 0.0])), Ray(ray_position, np.array([-5.0, 0.0, 0.0])), Ray(ray_position, np.array([-5.0, -1.0, 0.0]))])
        ray_arr[2] = np.array([Ray(ray_position, np.array([-5.0, 1.0, -1.0])), Ray(ray_position, np.array([-5.0, 0.0, -1.0])), Ray(ray_position, np.array([-5.0, -1.0, -1.0]))])

        for i in range(len(ray_arr)):
            for j in range(len(ray_arr[i])):
                ray_arr[i,j].direction = ray_arr[i,j].direction / norm(ray_arr[i,j].direction)

        camera = Camera(basis, TEST_SIZE, e, d)

        camera.change_camera_position(new_e)

        for i in range(len(ray_arr)):
            for j in range(len(ray_arr[i])):
                assertionVal = np.allclose(camera.rays[i,j].origin, ray_arr[i][j].origin) and np.allclose(camera.rays[i,j].direction, ray_arr[i][j].direction)
                self.assertTrue(assertionVal)
    
    def test_get_camera_type(self):
        TEST_SIZE = (3, 3)
        expected_camera_type_parallel = 'orthographic'
        expected_camera_type_perspective = 'perspective'

        u = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([1.0, 0.0, 0.0])

        e = np.array([0.0, 0.0, 0.0])

        d = 5.0

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        parallel_camera = Camera(basis, TEST_SIZE, e)
        actual_output = parallel_camera.get_camera_type()

        self.assertEqual(actual_output, expected_camera_type_parallel)

        perspective_camera = Camera(basis, TEST_SIZE, e, d)
        actual_output = perspective_camera.get_camera_type()

        self.assertEqual(actual_output, expected_camera_type_perspective)
    
    def test__get_sphere_valid_solution(self):
        TEST_SIZE = (1,1)

        u = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([1.0, 0.0, 0.0])

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        e = np.array([0.0, 0.0, 0.0])

        camera = Camera(basis, TEST_SIZE, e)

        center = np.array([-2.0, 0.0, 0.0])
        radius = 1.2
        color = np.array([0, 0, 0])
        phong_exponent = 1.0
        sphere = Sphere(center, color, radius, phong_exponent)

        expected_output = 0.8

        actual_output = camera._get_sphere_valid_solution(sphere, camera.rays[0][0])

        self.assertAlmostEqual(actual_output, expected_output)

    def test__get_triangle_valid_solution(self):
        # Triangles aren't supported yet
        pass

    def test__find_solutions(self):
        TEST_SIZE = (1,1)

        u = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])
        w = np.array([1.0, 0.0, 0.0])

        basis = {
            'u': u,
            'v': v,
            'w': w
        }

        e = np.array([0.0, 0.0, 0.0])

        camera = Camera(basis, TEST_SIZE, e)

        center = np.array([-2.0, 0.0, 0.0])
        radius = 1.2
        color = np.array([0, 0, 0])
        phong_exponent = 1.0
        sphere = Sphere(center, color, radius, phong_exponent)

        expected_output = np.empty(TEST_SIZE, dtype=object)
        expected_output[0,0] = (0.8, sphere)

        actual_output = camera._find_solutions([sphere])

        self.assertTrue(expected_output.shape == actual_output.shape)

        for i in range(expected_output.shape[0]):
            for j in range(expected_output.shape[1]):
                expected_value = expected_output[i, j]
                actual_value = actual_output[i, j]
                
                if expected_value is not None and actual_value is not None:
                    self.assertTrue(expected_value[1] == actual_value[1])
                    self.assertAlmostEqual(expected_value[0], actual_value[0])
                else:
                    self.assertTrue(expected_value == actual_value)

def test__calculate_colors(self): # TODO
    pass

if __name__ == '__main__':
    unittest.main()
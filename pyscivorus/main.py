import logging
import logging.config
import pygame as pg
import numpy as np
import logging_config # Although this isnt being used here it's running something

from scene import Scene

rootLogger = logging.getLogger('root')

SCREEN_DIMENSIONS = (1025, 511) # (width, height)
# For now only odd dimensions are supported
# Also changing dimensions in runtime is not supported

def main():
    rootLogger.info('PyGame initialized.')
    pg.init()
    screen = pg.display.set_mode(SCREEN_DIMENSIONS)
    pg.display.set_caption('Pyscivorus')
    program_is_alive = True
    surf = pg.Surface(SCREEN_DIMENSIONS)

    scene = Scene()
    scene.set_background_color(np.array([50, 0, 50]))
    scene.set_ambient_light_intensity(0.5)
    scene.add_directional_light(1, np.array([0, 0, -1]))
    scene.add_directional_light(0.5, np.array([1, -1, -1]))
    scene.add_sphere(np.array([1000, 700, 200]), np.array([0, 0, 150]), 100, 100)
    scene.add_sphere(np.array([200, 500, 300]), np.array([50, 50, 50]), 50)

    basis = {
        'u': np.array([0, -1, 0]),
        'v': np.array([0, 0, 1]),
        'w': np.array([-1, 0, 0])
    }

    scene.with_orthographic_camera(basis, np.array([0, SCREEN_DIMENSIONS[0]//2, SCREEN_DIMENSIONS[1]//2]), SCREEN_DIMENSIONS)

    rgb_arr = scene.take_picture()
    
    pg.surfarray.blit_array(surf, rgb_arr)

    rootLogger.info('Entering game loop.')
    # game loop
    while program_is_alive:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                rootLogger.info('User quit program.')
                program_is_alive = False

        screen.blit(surf, (0,0))
        pg.display.flip()
        
    pg.quit()
    rootLogger.info('PyGame quit.')
    exit()

if __name__ == '__main__':
    main()
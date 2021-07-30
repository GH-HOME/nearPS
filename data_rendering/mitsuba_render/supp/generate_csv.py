import numpy as np
import os
import glob

para_data_dir = './'

# Generate the shape
shape_dir = r'F:\dataset\shape_set\unit_sphape'
shape_list = glob.glob(os.path.join(shape_dir, '*.obj'))
shape_list_path = os.path.join(para_data_dir, 'shape_list.csv')
np.savetxt(shape_list_path, np.array(shape_list), fmt='%s',delimiter=',')

# Generate the light
def generate_LEDs(radius, numx, numy, z, light_ins):

    LEDxy = np.meshgrid(np.arange(-numx, numx+1), np.arange(-numy, numy+1))
    LEDz = np.ones([2 * numx + 1, 2 * numx + 1, 1]) * z
    LED_ins = np.ones([2 * numx + 1, 2 * numx + 1, 1]) * light_ins
    LEDxy = radius * np.array(LEDxy).transpose([1, 2, 0])
    LED_array = np.dstack([LEDxy, LEDz, LED_ins])
    return LED_array

LED_intervel = 0.2
LED_num_x, LED_num_y = 5, 5
LED_depth_plane = 0
light_ins = 100
LED_array = generate_LEDs(LED_intervel, LED_num_x, LED_num_y , LED_depth_plane, light_ins)
light_list_path = os.path.join(para_data_dir, 'light_list_intervel_{}_numx_{}_numy_{}_depth_{}.csv'
                               .format(LED_intervel, LED_num_x, LED_num_y, LED_depth_plane))
np.savetxt(light_list_path, np.array(LED_array).reshape(-1, 4), fmt='%f',delimiter=',')



# Generate the albedo set corresponding to the shape list
resolution = np.array([64, 64])
scale = np.array([1, 2, 4, 8, 10, 16, 32])
resolution_set = resolution[:, np.newaxis] @ scale[np.newaxis, :]
scale_list_path = os.path.join(para_data_dir, 'scale_list_{}_{}.csv'
                               .format(resolution[0], resolution[1]))
np.savetxt(scale_list_path, np.array(resolution_set).T, fmt='%d',delimiter=',')

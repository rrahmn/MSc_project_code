import numpy as np




fx = np.random.uniform(200,1600, 2000)

fy = np.random.uniform(np.where(fx - 300 > 200, fx - 300, 200),
                       np.where(fx + 300 < 1600, fx + 300, 1600))


width = 300
height = 300

cx = np.round(np.random.uniform(width/2 - 0.15*width/2, width/2 + 0.15*width/2, 2000)).astype(int)
cy = np.round(np.random.uniform(height/2 - 0.15*height/2, height/2 + 0.15*height/2, 2000)).astype(int)





intrinsic_data = np.array([fx, fy, cx, cy]).T

np.save('intrinsic_set_training.npy', intrinsic_data)
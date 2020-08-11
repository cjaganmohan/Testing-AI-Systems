#Reference URL - https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib

import matplotlib.pyplot as plt
import numpy as np

a = np.array([ [0.3387794, 0.   ,     0.,        0.,        0.,        0., 0.5496112],
               [0.       , 0.6757513, 0.       , 0.    ,    0.      ,  1.3618413, 2.1479137],
               [0.       , 0.       , 1.9722133, 0.    ,   0.      ,  0., 0.],
               [0.       , 0.       , 0.       , 0.    ,    0.      ,  0., 0.],
               [0.       , 0.       , 0.       , 0.    ,    0.      ,  0., 0.],
               [2.6597185, 0.       , 0.       , 0.    ,    2.418162,  0., 0.] ,
               [0.       , 0.       , 1.9358828, 0.    ,    0.      ,  0., 0.5255934]])
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()


b = np.array([ [0.       , 1.7375219,     0.,        0.,        0.138969,        0., 0.5496112],
               [0.       , 0.77372384, 0.      , 0.8295324 ,    0. ,  1.3618413, 0],
               [0.       , 3.9958048, 4.0298185, 0.    ,   0.      ,  0., 4.570733],
               [0.       , 0.       , 0.       , 0.    ,    0.      ,  0., 0.],
               [0.       , 0.       , 0.       , 0.    ,    0.      ,  0., 0.],
               [2.6597185, 0.       , 0.       , 0.    ,    2.418162,  0., 0.] ,
               [0.       , 0.       , 1.9358828, 0.    ,    0.      ,  0., 0.5255934]])

plt.imshow(b, cmap='hot', interpolation='nearest')
plt.show()

eucledian_distance = np.linalg.norm(a-a)
print(eucledian_distance)
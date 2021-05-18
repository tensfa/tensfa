img_path = '/path/to/my/image.jpg'

import numpy as np
from keras.preprocessing import image
x = image.load_img(img_path, target_size=(250, 250))

x = image.img_to_array(x)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
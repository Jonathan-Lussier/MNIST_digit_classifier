import numpy as np

f = open("MNIST/train-images.idx3-ubyte", 'rb')

image_size = 28
num_images = 4

try:
    print(f.read(16))
except UnicodeDecodeError as error:
    print(error)


buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
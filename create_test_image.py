import numpy as np
from PIL import Image

# Create a random RGB image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save("test_image.jpg")
print("test_image.jpg created")

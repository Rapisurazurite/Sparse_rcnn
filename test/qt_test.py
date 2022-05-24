import cv2
import numpy as np

empty_image = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imshow('empty_image', empty_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

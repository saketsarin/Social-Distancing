import cv2
import imutils
from People_Detection import Detect

img = cv2.imread('./testing/t1.jpg')
img = imutils.resize(img, width=700)
img = Detect(img)
result_image = img
cv2.imwrite('./testing/image_path_11.jpg', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
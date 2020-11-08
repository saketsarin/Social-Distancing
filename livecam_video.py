import cv2
from People_Detection import Detect

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./testing/live_video_1.mp4',fourcc, 60.0,(980,498))

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame,(980,498))
    frame = Detect(frame)
    out.write(frame)
    cv2.resizeWindow('Human Detection', 600, 600)
    q = cv2.waitKey(30) & 0xff
    if q == 27:
        break

cv2.destroyAllWindows()
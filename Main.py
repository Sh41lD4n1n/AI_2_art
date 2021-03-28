import cv2 as cv
import numpy as np
W = 512

def hexagon(img):
    l = 20
    for i in range(int(W*2/(l))):
        y0 = l*i/2
        if (i+1)%2==1:
            x0 = 0
        else:
            x0 = 3*l/4+l/10
        for j in range(int(4*W/(l*6))+1):
            ppt1 = np.array([[int(l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                        [int(3 * l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                        [x0 + l, y0 + l / 2],
                        [int(3 * l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                        [int(l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                        [x0, int(y0 + l / 2)]], np.int32)
            ppt1 = ppt1.reshape((-1, 1, 2))
            img = cv.fillConvexPoly(img, ppt1, (0, 255, 255), cv.LINE_4)
            x0 = 6 * l / 4+l/5+ x0
    return img

# Create black empty images
size = W, W, 3
atom_image = np.zeros(size, dtype=np.uint8)
hexagon(atom_image)
cv.imwrite('output1.jpg',atom_image)

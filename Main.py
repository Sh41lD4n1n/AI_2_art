#from skimage.io import imread
'''import cv2
import numpy as np
im = cv2.imread('lemur.jpg')
print(im.shape)
print(im[0,0,1])
im[:, :, (0,    1)] = 0
cv2.imwrite('output.jpg',im)'''
#img = imread('data/lemur.jpg')
import cv2 as cv
import numpy as np
W = 400

def hexagon(img):
    l = 100
    x0 = 0
    y0 = 0
    for i in range(int(4*W/(l*6))+1):
        ppt1 = np.array([[int(l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                    [int(3 * l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                    [x0 + l, y0 + l / 2],
                    [int(3 * l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                    [int(l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                    [x0, int(y0 + l / 2)]], np.int32)
        ppt1 = ppt1.reshape((-1, 1, 2))
        img = cv.fillConvexPoly(img, ppt1, (0, 255, 255), cv.LINE_4)
        x0 = 6 * l / 4+ x0
    #-----
    '''l = 100
    x0 = 6*l/4
    y0 = 0
    ppt2 = np.array([[int(l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                     [int(3 * l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                     [x0 + l, y0 + l / 2],
                     [int(3 * l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                     [int(l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                     [x0, int(y0 + l / 2)]], np.int32)
    ppt2 = ppt2.reshape((-1, 1, 2))
    img = cv.fillConvexPoly(img, ppt2, (0, 255, 255),cv.LINE_4)'''
    #----
    l = 100
    x0 = 3 * l / 4
    y0 = l / 2
    ppt3 = np.array([[int(l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                     [int(3 * l / 4 + x0), int(l * (-3 ** 0.5 + 2) / 4 + y0)],
                     [x0 + l, y0 + l / 2],
                     [int(3 * l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                     [int(l / 4 + x0), int(l * (3 ** 0.5 + 2) / 4 + y0)],
                     [x0, int(y0 + l / 2)]], np.int32)
    ppt3 = ppt3.reshape((-1, 1, 2))
    img = cv.fillConvexPoly(img, ppt3, (0, 255, 255), cv.LINE_4)
    return img


atom_window = "Drawing 1: Atom"
rook_window = "Drawing 2: Rook"
# Create black empty images
size = W, W, 3
atom_image = np.zeros(size, dtype=np.uint8)
rook_image = np.zeros(size, dtype=np.uint8)
hexagon(atom_image)



cv.imwrite('output1.jpg',atom_image)
cv.imwrite('output2.jpg',rook_image)

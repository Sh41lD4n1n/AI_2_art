import cv2 as cv
import numpy as np
import random
W = 512
class Picture:
    def __init__(self,pict_size,elem_size):
        self.elem_size = elem_size
        self.pict_size = pict_size

        self.picture = np.zeros((pict_size,pict_size,3), dtype=np.uint8)

        self.num_of_colomn = int(self.pict_size * 2 / (self.elem_size))
        self.num_of_row = int(4 * W / (self.elem_size * 6)) + 1

        # array of color of element
        self.color_array=np.random.randint(0,256,(self.num_of_colomn,self.num_of_row,3),dtype=np.int32)
        self.fill_image_with_hexagons()

    def fit_function(self,picture):
        if(picture.shape==self.picture.shape):
            result = (picture-self.picture)
            fit = int(np.sum(a=result*result)**0.5)
        else:
            fit = 0
        #print(fit)
        return fit
    def mutation(self):
        number_of_modification = int(self.num_of_row*self.num_of_colomn*0.10)
        for i in range(number_of_modification):
            x = random.randint(0,self.num_of_colomn-1)
            y = random.randint(0, self.num_of_row-1)
            self.color_array[x][y]=np.random.randint(0,256,(3))
        self.fill_image_with_hexagons()

    def crossover(Pict, otherPict):
        chrom_size = int(Pict.num_of_row * Pict.num_of_colomn)
        p1 = random.randint(0, chrom_size - 1)
        p2 = random.randint(0, chrom_size - 1)
        if p1 > p2:
            temp_p = p1
            p1 = p2
            p2 = temp_p
        for i in range(p1, p2):
            temp = np.array([Pict.color_array.reshape((chrom_size, 3))[i][0],
                             Pict.color_array.reshape((chrom_size, 3))[i][1],
                             Pict.color_array.reshape((chrom_size, 3))[i][2]], dtype=np.int32)
            Pict.color_array.reshape((chrom_size, 3))[i][0] = otherPict.color_array.reshape((chrom_size, 3))[i][0]
            Pict.color_array.reshape((chrom_size, 3))[i][1] = otherPict.color_array.reshape((chrom_size, 3))[i][1]
            Pict.color_array.reshape((chrom_size, 3))[i][2] = otherPict.color_array.reshape((chrom_size, 3))[i][2]

            otherPict.color_array.reshape((chrom_size, 3))[i][0] = temp[0]
            otherPict.color_array.reshape((chrom_size, 3))[i][1] = temp[1]
            otherPict.color_array.reshape((chrom_size, 3))[i][2] = temp[2]
        Pict.fill_image_with_hexagons()
        otherPict.fill_image_with_hexagons()

    def show(self):
        cv.imwrite('output1.jpg', self.picture)

    def fill_image_with_hexagons(self):
        for i in range(self.num_of_colomn):
            y0 = self.elem_size * i / 2
            if (i + 1) % 2 == 1:
                x0 = 0
            else:
                x0 = 3 * self.elem_size / 4 + self.elem_size / 10
            for j in range(self.num_of_row):
                ppt1 = np.array([[int(self.elem_size / 4 + x0), int(self.elem_size * (-3 ** 0.5 + 2) / 4 + y0)],
                                 [int(3 * self.elem_size / 4 + x0), int(self.elem_size * (-3 ** 0.5 + 2) / 4 + y0)],
                                 [x0 + self.elem_size, y0 + self.elem_size / 2],
                                 [int(3 * self.elem_size / 4 + x0), int(self.elem_size * (3 ** 0.5 + 2) / 4 + y0)],
                                 [int(self.elem_size / 4 + x0), int(self.elem_size * (3 ** 0.5 + 2) / 4 + y0)],
                                 [x0, int(y0 + self.elem_size / 2)]], np.int32)
                ppt1 = ppt1.reshape((-1, 1, 2))
                c1,c2,c3=self.color_array[i][j]
                self.picture= cv.fillConvexPoly(self.picture, ppt1, (int(c1), int(c2), int(c3)), cv.LINE_4)
                x0 = 6 * self.elem_size / 4 + self.elem_size / 5 + x0

def crossover(Pict,otherPict):
    chrom_size = int(Pict.num_of_row * Pict.num_of_colomn)
    p1 = random.randint(0, chrom_size - 1)
    p2 = random.randint(0, chrom_size - 1)
    if p1>p2:
        temp_p=p1
        p1 = p2
        p2=temp_p
    for i in range(p1,p2):
        temp = np.array([Pict.color_array.reshape((chrom_size,3))[i][0],
                         Pict.color_array.reshape((chrom_size,3))[i][1],
                         Pict.color_array.reshape((chrom_size,3))[i][2]],dtype=np.int32)
        Pict.color_array.reshape((chrom_size, 3))[i][0] = otherPict.color_array.reshape((chrom_size, 3))[i][0]
        Pict.color_array.reshape((chrom_size, 3))[i][1] = otherPict.color_array.reshape((chrom_size, 3))[i][1]
        Pict.color_array.reshape((chrom_size, 3))[i][2] = otherPict.color_array.reshape((chrom_size, 3))[i][2]

        otherPict.color_array.reshape((chrom_size, 3))[i][0] = temp[0]
        otherPict.color_array.reshape((chrom_size, 3))[i][1] = temp[1]
        otherPict.color_array.reshape((chrom_size, 3))[i][2] = temp[2]
    Pict.fill_image_with_hexagons()
    otherPict.fill_image_with_hexagons()

# Create black empty images
size = W, W, 3
img = cv.imread('input.jpg')
cv.imwrite('output2.jpg',img)
chromosome1 = Picture(W,15)
chromosome1.fit_function(img)
chromosome2 = Picture(W,15)
chromosome2.fit_function(img)
cv.imwrite('output1.jpg',chromosome1.picture)
cv.imwrite('output2.jpg',chromosome2.picture)
input()
chromosome1.crossover(chromosome2)
cv.imwrite('output1.jpg',chromosome1.picture)
cv.imwrite('output2.jpg',chromosome2.picture)
#atom_image = np.zeros(size, dtype=np.uint8)
#hexagon(atom_image)

#cv.imwrite('output1.jpg',atom_image)

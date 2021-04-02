import cv2 as cv
import numpy as np
import random
from skimage.metrics  import structural_similarity as ssim
W = 512
class Picture:
    def __init__(self,pict_size,elem_size):
        self.elem_size = elem_size
        self.pict_size = pict_size

        self.picture = np.zeros((pict_size,pict_size,3), dtype=np.uint8)

        self.num_of_colomn = int(self.pict_size * 2 / (self.elem_size))
        self.num_of_row = int(4 * W / (self.elem_size * 6)) + 1

        # array of color of element
        self.color_array=np.full((self.num_of_colomn,self.num_of_row,3),255) #np.random.randint(0,256,(self.num_of_colomn,self.num_of_row,3),dtype=np.int32)
        self.fill_image_with_hexagons()
        self.fit_value=0
    def __del__(self):
        del self.picture
        del self.color_array
    def fit_function(self,picture):
        if(picture.shape==self.picture.shape):
            self.fit_value = ssim(picture,self.picture,multichannel=True)#np.linalg.norm(np.absolute(picture-self.picture))
        else:
            self.fit_value = 0
    def mutation(self):
        number_of_modification = int(self.num_of_row*self.num_of_colomn*0.10)
        for i in range(number_of_modification):
            x = random.randint(0,self.num_of_colomn-1)
            y = random.randint(0, self.num_of_row-1)
            self.color_array[x][y]=np.random.randint(0,256,(3))
        self.fill_image_with_hexagons()

    def crossover(self, otherPict):
        child1 = Picture(self.pict_size,self.elem_size)
        child2 = Picture(self.pict_size, self.elem_size)

        chrom_size = int(self.num_of_row * self.num_of_colomn)
        p1 = random.randint(0, chrom_size - 1)
        p2 = random.randint(0, chrom_size - 1)
        if p1 > p2:
            temp_p = p1
            p1 = p2
            p2 = temp_p
        for i in range(p1):
            child2.color_array.reshape((chrom_size, 3))[i][0] = otherPict.color_array.reshape((chrom_size, 3))[i][0]
            child2.color_array.reshape((chrom_size, 3))[i][1] = otherPict.color_array.reshape((chrom_size, 3))[i][1]
            child2.color_array.reshape((chrom_size, 3))[i][2] = otherPict.color_array.reshape((chrom_size, 3))[i][2]

            child1.color_array.reshape((chrom_size, 3))[i][0] = self.color_array.reshape((chrom_size, 3))[i][0]
            child1.color_array.reshape((chrom_size, 3))[i][1] = self.color_array.reshape((chrom_size, 3))[i][1]
            child1.color_array.reshape((chrom_size, 3))[i][2] = self.color_array.reshape((chrom_size, 3))[i][2]
        for i in range(p1, p2):
            child1.color_array.reshape((chrom_size, 3))[i][0] = otherPict.color_array.reshape((chrom_size, 3))[i][0]
            child1.color_array.reshape((chrom_size, 3))[i][1] = otherPict.color_array.reshape((chrom_size, 3))[i][1]
            child1.color_array.reshape((chrom_size, 3))[i][2] = otherPict.color_array.reshape((chrom_size, 3))[i][2]

            child2.color_array.reshape((chrom_size, 3))[i][0] = self.color_array.reshape((chrom_size, 3))[i][0]
            child2.color_array.reshape((chrom_size, 3))[i][1] = self.color_array.reshape((chrom_size, 3))[i][1]
            child2.color_array.reshape((chrom_size, 3))[i][2] = self.color_array.reshape((chrom_size, 3))[i][2]
        for i in range(p2):
            child2.color_array.reshape((chrom_size, 3))[i][0] = otherPict.color_array.reshape((chrom_size, 3))[i][0]
            child2.color_array.reshape((chrom_size, 3))[i][1] = otherPict.color_array.reshape((chrom_size, 3))[i][1]
            child2.color_array.reshape((chrom_size, 3))[i][2] = otherPict.color_array.reshape((chrom_size, 3))[i][2]

            child1.color_array.reshape((chrom_size, 3))[i][0] = self.color_array.reshape((chrom_size, 3))[i][0]
            child1.color_array.reshape((chrom_size, 3))[i][1] = self.color_array.reshape((chrom_size, 3))[i][1]
            child1.color_array.reshape((chrom_size, 3))[i][2] = self.color_array.reshape((chrom_size, 3))[i][2]
        child1.fill_image_with_hexagons()
        child2.fill_image_with_hexagons()
        return child1,child2

    def show(self,file_name):
        cv.imwrite(file_name, self.picture)

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

#[mem1...]
def selection(population):
    winner=[]
    losers=[]
    while len(population)>0:
        mem1 = random.randint(0, len(population)-1)
        mem2 = (random.randint(0, len(population)-1)+mem1)%len(population)
        if (population[mem1].fit_value>population[mem2].fit_value):
            winner.append(population[mem1])
            losers.append(population[mem2])
        else:
            winner.append(population[mem2])
            losers.append(population[mem1])
        population.pop(mem1)
        population.pop(mem2-1)
    return winner,losers
def algorithm(population,image):
    #check fit function
    min = 25
    imin = 0
    gener_number=0
    total_gener_number = 4000
    while (min>0.8 or gener_number<total_gener_number):
        print(f"generation{gener_number}")
        gener_number = gener_number+1

        for i in range(len(population)):
            population[i].fit_function(image)

        part1,remove_items=selection(population)
        for i in range(len(remove_items)):
            elem = remove_items.pop(0)
            del elem
        del remove_items
        #perform crossover
        for i in range(1,len(part1),2):
            c1,c2=part1[i].crossover(part1[i - 1])
            part1.append(c1)
            part1.append(c2)
        #mutation
        population,for_mutation=selection(part1)
        for i in range(len(part1)):
            elem = part1.pop(0)
            del elem
        del part1
        for i in range(len(for_mutation)):
            for_mutation[i].mutation()
            population.append(for_mutation[i])
        del for_mutation
        #show ten images
        for i in range(10):
            population[i].show(f"output{i}.jpg")

        min = population[0].fit_value
        imin = 0
        for i in range(1, len(population)):
            fit = population[i].fit_value
            if (fit > min):
                min = fit
                imin = i
        print(min)
        population[imin].show("best_in_pop.jpg")
    for i in range(len(population)):
        fit = population[i].fit_value
        print(fit)
        population[i].show(f"output{i}.jpg")
    population[imin].show("midle.jpg")



# Create black empty images
size = W, W, 3
img = cv.imread('input2.jpg')
population = []
#fit function calculated in some cases
for i in range(100):
    population.append(Picture(W,15))
#chromosome1 = Picture(W,15)
algorithm(population,img)
#cv.imwrite('outputf.jpg',outputim)

#atom_image = np.zeros(size, dtype=np.uint8)
#hexagon(atom_image)

#cv.imwrite('output1.jpg',atom_image)

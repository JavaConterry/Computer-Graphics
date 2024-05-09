import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

class Interpolatable_Image:

    def __init__(self, src='img.png'):
        image = Image.open(src).convert("L")
        self.img = np.array(image)
        self.img_y, self.img_x = self.img.shape
        self.multiply = None


    def __expand(self, multiply=2): #can be improved by custom size
        if(self.multiply!=multiply):
            self.multiply = multiply
        new_img = np.zeros(shape=(self.img_y*self.multiply-1, self.img_x*self.multiply-1))
        for i in range(0, self.img_y):
            for j in range(0, self.img_x):
                new_img[i*self.multiply][j*self.multiply] = self.img[i][j]
        # self.show(new_img)
        return new_img
    

    def show(self, spec=[]):
        if(spec.shape != (1,0)):
            Image.fromarray(spec).show()    
        else: Image.fromarray(self.img).show()

    
    def liniar_interpolation(self, scale=2):
        new_img = self.__expand(scale)
        for i in range(0, len(new_img)-self.multiply, self.multiply):
            for j in range(0, len(new_img[0])-self.multiply, self.multiply):
                X1 = j; X2 = j+self.multiply
                Y1 = i; Y2 = i+self.multiply
                for y in range(Y1, Y2+1):
                    for x in range(X1, X2+1):
                        temp = self.img[int(Y1/self.multiply)][int(X1/self.multiply)]*(X2-x)*(Y2-y)+self.img[int(Y1/self.multiply)][int(X2/self.multiply)]*(x-X1)*(Y2-y)+self.img[int(Y2/self.multiply)][int(X1/self.multiply)]*(X2-x)*(y-Y1)+self.img[int(Y2/self.multiply)][int(X2/self.multiply)]*(x-X1)*(y-Y1)
                        res = temp/((X2-X1)*(Y2-Y1))
                        new_img[y][x] = res
        return new_img


    def furie_interpolation(self):
        new_img = self.img.copy()
        nx,ny = 100,200
        x, y = [i for i in range(nx)], [i for i in range(ny)]
        put_x = 3; put_y = 3
        X = [0]*nx
        Y = [0]*ny
        for i in range(nx):
            X[i] = i + (i-1)*put_x
            for j in range(ny):
                Y[i] = j+(j-1)*put_y
                new_img[i][j] = math.sin(x[i]/nx*math.pi)*math.cos(2*y[j]/ny*math.pi)+1
        Y, X = np.meshgrid(Y, X)
        nx_new = nx+(nx-1)*put_x
        ny_new = ny+(ny-1)*put_y
        x_new = [i for i in range(nx_new)]
        y_new = [i for i in range(ny_new)]
        y_new, x_new = np.meshgrid(y_new, x_new)
        my_image_new = zoom(new_img, (nx_new/new_img.shape[0], ny_new/new_img.shape[1]))        
        return my_image_new




inter = Interpolatable_Image('img.png')


# lin_int = inter.liniar_interpolation(scale=2)
# Image.fromarray(lin_int).show()

# fur = inter.furie_interpolation()
# Image.fromarray(fur).show()


def poly(xs, ys, show=True):
    ys = np.array(ys).reshape((len(ys), 1))
    print('ys:', ys)
    xs_poly = np.array([[x**i for i in range(len(xs)-1, -1, -1)] for x in xs])
    print('xs_poly:', xs_poly)
    res_coefs = np.linalg.inv(xs_poly).dot(ys)
    res_coefs = [val for val in res_coefs.reshape((1, len(res_coefs)))[0]]
    print('result:', res_coefs)

    #show
    if(show):
        def polynomial(x, coef):
            y = sum(coef * x**i for i, coef in enumerate(coef[::-1]))
            return y

        freq = 3
        x_values = np.linspace(min(xs), max(xs), freq*len(xs)) #the predicted values
        y_values = polynomial(x_values, res_coefs)

        plt.plot(x_values, y_values, label='Polynomial')
        plt.scatter(xs, ys, color='red', label='Data Points') 
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Polynomial Fit')
        plt.legend()
        plt.grid(True)
        plt.show()


x_data = [0, 2, 4, 6]
y_data = [0, 4, 8, 20]

poly(x_data,y_data)

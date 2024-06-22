# %%
import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2*tile_size) == 0:
            self.resolution = resolution
            self.tile_size = tile_size
            self.output = np.ndarray
        else:
            print("Error: object not created! Resolution ", end="")
            print(resolution, end="")
            print(" and tile size ", end="")
            print(tile_size, end="")
            print(" may produce truncated patterns. Plesae try again.")

    def draw(self):
        # make kernel tile
        bKern = np.zeros((self.tile_size, self.tile_size))
        wKern = np.ones((self.tile_size, self.tile_size))

        ### Using h/vstack ###
        # topHalf = np.hstack((bKern, wKern))
        # botHalf = np.hstack((wKern, bKern))
        # kernelTile = np.vstack((topHalf, botHalf))

        kernelTile = np.bmat([[bKern,wKern], [wKern, bKern]])

        # fill out the board
        tilingSize = int(self.resolution / (2 * self.tile_size))
        self.output = np.tile(kernelTile, (tilingSize, tilingSize))

        copy = np.copy(self.output)
        
        return copy

    def show(self):
        plt.imshow(self.output, cmap = 'gray')

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

        self.output = np.ndarray

    def draw(self):
        # black canvas
        canvas = np.zeros((self.resolution, self.resolution))

        # meshgrid for drawing
        x_axis = np.arange(0, self.resolution)
        # y_axis = np.arange(self.resolution-1, -1, -1) # to make 0,0 on the bottom left (quirk of meshgrid)
        y_axis = np.arange(0, self.resolution)
        x, y = np.meshgrid(x_axis, y_axis, sparse = 0)

        # formula for circle in euclidean plane is r**2 = (x-h)**2 + (y-k)**2
        # where h and k are the cirlce's center
        r = np.sqrt((x - self.position[0])**2 + (y - self.position[1])**2)

        # x and y coordinates where the circle exists
        circle_x, circle_y = np.where(r <= self.radius)

        # make them white on the canvas
        canvas[circle_x, circle_y] = 1

        self.output = canvas

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap = 'gray', extent = [0, self.resolution, self.resolution, 0])

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.array

    def draw(self):
        # black canvas
        canvas = np.zeros((self.resolution, self.resolution, 3))

        # meshgrid for easy plotting
        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        x_axis, y_axis = np.meshgrid(x, y, sparse = 0)

        # RBG gradients
        red_gradiant = np.linspace(0, 1, self.resolution)
        green_gradiant = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
        blue_gradiant = np.linspace(1, 0, self.resolution)

        # apply gradients to canvas
        canvas[:,:,0] = red_gradiant
        canvas[:,:,1] = green_gradiant
        canvas[:,:,2] = blue_gradiant

        self.output = canvas

        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)


# %%
#if __name__ == "__main__":
#    #print("Hello World!")
#    c = Spectrum(255)
#    c.draw()
#    #print("Output from main function:")
#    #print(c.output)
#    c.show()

# %%

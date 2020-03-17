'''2D grid graph of an image, cvx algorithms comparision '''
import time
import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from snap import *
from solver import solve_GroupFL
from solver_pool import solve_GroupFL_pool
from NetworkLasso_solver_Hallac import runADMM
from NetworkLasso_solver import solve_NetworkLasso



def img():
    ''' group fused lasso algorithm with explicit updates '''
    # img = mpimg.imread('Data/ABC.png')
    img = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            if (i - 4)**2 + (j - 4)**2 < 5:
                img[i][j] = np.array([0.5, 0.5, 0.5])

    imgsize1, imgsize2, _ = img.shape
    # img=img.astype(float)/255
    np.random.seed(1)
    noise = 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    imgnoise = img + noise
    data = np.reshape(imgnoise, (-1, 3))

    plt.switch_backend('TkAgg')
    # img_recovered = np.reshape(x_hat, (imgsize1, imgsize2, 3))
    #
    fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(imgnoise)
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(img_recovered)
    # plt.savefig(
    #     'Figures/GridGraph_cvx/img', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    img()

'''2D grid graph of an image, explicit algorithms comparision '''
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from solver import solve_GroupFL
from NetworkLasso_solver import solve_NetworkLasso
# from solver_pool import solve_GroupFL_pool

from libGroupFL import coloredge, decompose_graph
from solver import solve_GroupFL
from solver_pool import solve_GroupFL_pool
from NetworkLasso_solver_Hallac import runADMM
# from NetworkLasso_solver import solve_NetworkLasso


def image_show():
    ''' print original image and image recovered'''
    img = mpimg.imread('Data/ABC.png')
    imgsize1, imgsize2, _ = img.shape
    # img=img.astype(float)/255
    noise = 0.1 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    imgnoise = img + noise
    data = np.reshape(imgnoise, (-1, 3))
    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    x_hat, _, _, _ = solve_NetworkLasso(
        data, graph, maxsteps=100, rho=8, lam=0.2)

    img_recovered = np.reshape(x_hat, (imgsize1, imgsize2, 3))

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.imshow(imgnoise)
    fig.add_subplot(1, 3, 3)
    plt.imshow(img_recovered)
    plt.savefig(
        'Figures/GridGraph/img_comparision', bbox_inches='tight')


def gfl_explicit_1():
    ''' group fused lasso algorithm with explicit updates '''
    # img = mpimg.imread('Data/ABC.png')
    # img = np.ones((10,10,3))
    img = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            if (i - 4)**2 + (j - 4)**2 < 5:
                img[i][j] = np.array([0.5, 0.5, 0.5])

    imgsize1, imgsize2, _ = img.shape
    # img=img.astype(float)/255
    noise = 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    imgnoise = img + noise
    data = np.reshape(imgnoise, (-1, 3))

    nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    temp_edges = []
    for k in range(0, imgsize1*imgsize2):
        for j in graph.neighbors(k):
            if j == k + 1:
                temp_edges.append([k, j])

    graph0_edges = []
    graph0_edges.append(temp_edges[0])
    for edge in temp_edges:
        if (edge[0] != graph0_edges[-1][1]) and (edge!=graph0_edges[-1]):
            graph0_edges.append(edge)

    graph1_edges = [edge for edge in list(graph.edges()) if edge not in graph0_edges]
    graph1_edges.sort()

    graph0_nodes = np.unique(np.array(graph0_edges))
    graph1_nodes = np.unique(np.array(graph1_edges))

    graph0 = nx.Graph()
    graph1 = nx.Graph()
    graph0.add_nodes_from(graph0_nodes)
    graph1.add_nodes_from(graph1_nodes)
    graph0.add_edges_from(graph0_edges)
    graph1.add_edges_from(graph1_edges)

    runtime = []
    objerror = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [32]

    _, obj, _, _ = solve_GroupFL(y=data, G=graph, G0=graph0, G1=graph1, maxsteps=10000, rho=4, lam=0.5)
    optobj = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        _, obj, _, tevolution = solve_GroupFL(y=data, G=graph, G0=graph0, G1=graph1, maxsteps=3000, rho=rho, lam=0.5)
        print('running time GraphLasso_explicit:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)

    # plotting
    plt.figure()
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.plot(runtime[1], objerror[1], label=r'$\rho=2^1$')
    plt.plot(runtime[2], objerror[2], label=r'$\rho=2^2$')
    plt.plot(runtime[3], objerror[3], label=r'$\rho=2^3$')
    plt.plot(runtime[4], objerror[4], label=r'$\rho=2^4$')
    plt.plot(runtime[5], objerror[5], label=r'$\rho=2^5$')
    plt.plot(runtime[6], objerror[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 20))
    plt.ylim((1e-12, 1e1))
    # plt.title('Graph Fused Lasso Algorithm (explict)')
    plt.title(r'Graph Fused Lasso Algorithm ' r'(explicit, $\lambda=0.5$)', fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    # plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/GridGraph/GraphFL_lamb=05_grid_graph_explicit', bbox_inches='tight')


def gfl_explicit_2():
    ''' group fused lasso algorithm with explicit updates '''
    img = mpimg.imread('Data/ABC.png')
    imgsize1, imgsize2, _ = img.shape
    # img=img.astype(float)/255
    np.random.seed(1)
    noise = 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    imgnoise = img + noise
    data = np.reshape(imgnoise, (-1, 3))

    nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    temp_edges = []
    for k in range(0, imgsize1*imgsize2):
        for j in graph.neighbors(k):
            if j == k + 1:
                temp_edges.append([k, j])

    graph0_edges = []
    graph0_edges.append(temp_edges[0])
    for edge in temp_edges:
        if (edge[0] != graph0_edges[-1][1]) and (edge!=graph0_edges[-1]):
            graph0_edges.append(edge)

    graph_edges = list(graph.edges())
    graph_edges.sort()
    graph1_edges = [edge for edge in list(
        graph.edges()) if edge not in graph0_edges]

    graph1_edges.sort()

    graph0_nodes = np.unique(np.array(graph0_edges))
    graph1_nodes = np.unique(np.array(graph1_edges))

    graph = nx.Graph()
    graph0 = nx.Graph()
    graph1 = nx.Graph()
    graph0.add_nodes_from(graph0_nodes)
    graph1.add_nodes_from(graph1_nodes)
    graph0.add_edges_from(graph0_edges)
    graph1.add_edges_from(graph1_edges)
    graph.add_nodes_from(range(imgsize1*imgsize2))
    graph.add_edges_from(graph_edges)

    objerror = []
    runtime = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [16]
    # _, obj, _, _ = solve_GroupFL(
        # y=data, G=graph, G0=graph0, G1=graph1, maxsteps=8000, rho=4, lam=10)

    # optobj = obj[-1]
    optobj = 286.88605869558523

    # print(optobj)
    # return

    for rho in rho_set:
        clock_time = time.time()
        _, obj, _, tevolution = solve_GroupFL(
            y=data, G=graph, G0=graph0, G1=graph1, maxsteps=5000, rho=rho, lam=10)
        print('running time GraphLasso_explicit:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)

    # plotting
    plt.figure()
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.plot(runtime[1], objerror[1], label=r'$\rho=2^1$')
    plt.plot(runtime[2], objerror[2], label=r'$\rho=2^2$')
    plt.plot(runtime[3], objerror[3], label=r'$\rho=2^3$')
    plt.plot(runtime[4], objerror[4], label=r'$\rho=2^4$')
    plt.plot(runtime[5], objerror[5], label=r'$\rho=2^5$')
    plt.plot(runtime[6], objerror[6], label=r'$\rho=2^6$')
    plt.legend()
    plt.xlim((0, 250))
    plt.ylim((1e-12, 1e2))
    plt.title('Graph Fused Lasso Algorithm (explict)')
    plt.xlabel('Running time (seconds)', fontsize=12)
    # plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/GridGraph/GraphFL_lamb=10_grid_graph_explicit', bbox_inches='tight')


def network_lasso_explicit_1():
    ''' network lasso algorithm with explicit updates '''
    # img = mpimg.imread('Data/ABC.png')
    # img = np.ones((10,10,3))
    img = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            if (i - 4)**2 + (j - 4)**2 < 5:
                img[i][j] = np.array([0.5, 0.5, 0.5])

    imgsize1, imgsize2, _ = img.shape
    np.random.seed(1)
    img_noise = img + 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    data = np.reshape(img_noise, (-1, 3))
    # nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    objerror = []
    runtime = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [16]
    _, obj, _, _ = solve_NetworkLasso(
        data, graph, maxsteps=10000, rho=2**6, lam=0.5)
    optobj = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        _, obj, _, tevolution = solve_NetworkLasso(
            data, graph, maxsteps=3000, rho=rho, lam=0.5)
        print('running time network_lasso_explicit:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)

    # plotting
    plt.figure()
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.plot(runtime[1], objerror[1], label=r'$\rho=2^1$')
    plt.plot(runtime[2], objerror[2], label=r'$\rho=2^2$')
    plt.plot(runtime[3], objerror[3], label=r'$\rho=2^3$')
    plt.plot(runtime[4], objerror[4], label=r'$\rho=2^4$')
    plt.plot(runtime[5], objerror[5], label=r'$\rho=2^5$')
    plt.plot(runtime[6], objerror[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 20))
    plt.ylim((1e-12, 1e1))
    # plt.title('Network Lasso Algorithm (explict)')
    plt.title(r'Network Lasso Algorithm ' r'(explicit, $\lambda=0.5$)', fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    # plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/GridGraph/Network_lamb=05_grid_graph_explicit', bbox_inches='tight')


def network_lasso_explicit_2():
    ''' network lasso algorithm with explicit updates '''
    img = mpimg.imread('Data/ABC.png')
    imgsize1, imgsize2, _ = img.shape
    img_noise = img + 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    data = np.reshape(img_noise, (-1, 3))
    # nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    objerror = []
    runtime = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [16]
    _, obj, _, _ = solve_NetworkLasso(
        data, graph, maxsteps=5000, rho=32, lam=10)
    optobj = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        _, obj, _, tevolution = solve_NetworkLasso(
            data, graph, maxsteps=1500, rho=rho, lam=10)
        print('running time network_lasso_explicit:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)

    # plotting
    plt.figure()
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.plot(runtime[1], objerror[1], label=r'$\rho=2^1$')
    plt.plot(runtime[2], objerror[2], label=r'$\rho=2^2$')
    plt.plot(runtime[3], objerror[3], label=r'$\rho=2^3$')
    plt.plot(runtime[4], objerror[4], label=r'$\rho=2^4$')
    plt.plot(runtime[5], objerror[5], label=r'$\rho=2^5$')
    plt.plot(runtime[6], objerror[6], label=r'$\rho=2^6$')
    plt.legend()
    plt.xlim((0, 250))
    plt.ylim((1e-12, 1e2))
    plt.title('Network Lasso Algorithm (explict)')
    plt.xlabel('Running time (seconds)',fontsize=12)
    # plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/GridGraph/Network_lamb=10_grid_graph_explicit', bbox_inches='tight')


if __name__ == "__main__":
    # image_show()
    gfl_explicit_1()
    # gfl_explicit_2()
    # network_lasso_explicit_1()
    # network_lasso_explicit_2()

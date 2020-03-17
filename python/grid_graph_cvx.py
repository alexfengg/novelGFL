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


def obj_find():
    ''' find optimal objective vlaue '''

    # img = mpimg.imread('Data/ABC.png')
    # imgsize1, imgsize2, _ = img.shape
    img = np.ones(16, 16, 3)

    np.random.seed(1)
    noise = 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    imgnoise = img + noise
    data = np.reshape(imgnoise, (-1, 3))

    nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)
    # print(list(graph.edges()))

    temp_edges = []
    for k in range(0, imgsize1 * imgsize2):
        for j in graph.neighbors(k):
            if j == k + 1:
                temp_edges.append([k, j])

    graph0_edges = []
    graph0_edges.append(temp_edges[0])
    for edge in temp_edges:
        if (edge[0] != graph0_edges[-1][1]) and (edge != graph0_edges[-1]):
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
    graph.add_nodes_from(range(imgsize1 * imgsize2))
    graph.add_edges_from(graph_edges)

    # print(list(graph.edges()))

    _, obj, _, _ = solve_GroupFL(
        y=data, G=graph, G0=graph0, G1=graph1, maxsteps=1000, rho=2, lam=10)

    optobj = obj[-1]
    print(obj)

    _, obj, _, _ = solve_NetworkLasso(
        y=data, G=graph, maxsteps=1000, rho=2, lam=10)
    optobj = obj[-1]
    print(obj)

    return
    plt.switch_backend('TkAgg')
    plt.figure()
    plt.plot(obj - obj[-1])
    # plt.ylim((1e-4, 1e3))
    plt.title('Graph Fused Lasso Algorithm (cvx)')
    plt.ylabel('error')
    plt.yscale('log')
    plt.show()

    with open('Figures/GridGraph_cvx/optobj.csv', 'w', newline='') as csvfile:
        objwriter = csv.writer(csvfile, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        objwriter.writerow([optobj] + ['lambda:'] + [10])


def gfl_cvx_1():
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

    nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    graph0_edges = []
    for k in range(0, nodes, 2):
        for j in graph.neighbors(k):
            if j == k + 1:
                graph0_edges.append([k, j])

    graph1_edges = [edge for edge in list(
        graph.edges()) if edge not in graph0_edges]

    graph1_edges.sort()

    # print(graph1_edges)

    # return

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
    # rho_set = [2]

    _, obj, _, _ = solve_GroupFL(
        y=data, G=graph, G0=graph0, G1=graph1, maxsteps=4500, rho=8, lam=0.5)

    optobj = obj[-1]
    print(optobj)

    # with open('Figures/GridGraph_cvx/optobj.csv', newline='') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for value in spamreader:
    #         obj = value

    for rho in rho_set:
        clock_time = time.time()
        x_hat, tevolution, obj, _ = solve_GroupFL_pool(
            y=data, G=graph, G0=graph0, G1=graph1, maxsteps=800, rho=rho, lam=0.5)
        print('GraphLasso_cvx:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        # objerror.append([i - obj[-1] for i in obj])
        runtime.append(tevolution)
        # print(obj)
        # print('obj:',obj)
        # print('time:',runtime)

    # print('obj_cvx:', obj)

    # img_recovered = np.reshape(x_hat, (imgsize1, imgsize2, 3))
    #
    # fig = plt.figure()
    # fig.add_subplot(1, 3, 1)
    # plt.imshow(img)
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(imgnoise)
    # fig.add_subplot(1, 3, 3)
    # plt.imshow(img_recovered)
    # plt.savefig(
    #     'Figures/GridGraph_cvx/img', bbox_inches='tight')

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
    plt.xlim((0, 120))
    plt.ylim((1e-6, 1e1))
    # plt.title('Graph Fused Lasso Algorithm (cvx)')
    plt.title(r'Graph Fused Lasso Algorithm ' r'(cvx, $\lambda=0.5$)', fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    # plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/GridGraph_cvx/GraphFL_lamb=05_grid_graph_cvx', bbox_inches='tight')


def network_lasso_cvx_1():
    ''' network lasso algorithm with explicit updates '''
    # img = mpimg.imread('Data/ABC.png')
    # img = np.ones((10, 10, 3))
    img = np.zeros((10, 10, 3))
    for i in range(10):
        for j in range(10):
            if (i - 4)**2 + (j - 4)**2 < 5:
                img[i][j] = np.array([0.5, 0.5, 0.5])

    imgsize1, imgsize2, _ = img.shape
    np.random.seed(1)
    img_noise = img + 0.2 * np.random.normal(0, 1, (imgsize1, imgsize2, 3))
    data = np.reshape(img_noise, (-1, 3))
    y = np.transpose(data)
    nodes = imgsize1 * imgsize2
    # graph settings

    graph = nx.grid_2d_graph(imgsize1, imgsize2)
    graph = nx.convert_node_labels_to_integers(graph)

    graph_edges = list(graph.edges())
    graph_edges.sort()
    # print(graph_edges)
    # return 0

    G = TUNGraph.New()
    for k in range(nodes):
        G.AddNode(k)

    for edge in graph_edges:
        G.AddEdge(edge[0], edge[1])

    print('Number of Edges:', G.GetEdges())

    edge_weights = TIntPrFltH()
    for edge in G.Edges():
        temp = TIntPr(edge.GetSrcNId(), edge.GetDstNId())
        edge_weights.AddDat(temp, 1)

    # Initialize variables to 0
    x_initial = np.zeros((3, nodes))
    u_initial = np.zeros((3, 2 * G.GetEdges()))
    z_initial = np.zeros((3, 2 * G.GetEdges()))

    objerror = []
    runtime = []
    rho_set = [2**k for k in range(7)]
    # rho_set=[1]
    # optobj = 287.905779648668

    _, obj, _, _ = solve_NetworkLasso(
        y=data, G=graph, maxsteps=1500, rho=32, lam=0.5)

    optobj = obj[-1]

    # optobj3 = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        x_hat, tevolution, obj, _ = runADMM(G1=G, sizeOptVar=3, sizeData=3, lamb=0.5, rho=rho, numiters=300, x=x_initial,
                                            u=u_initial, z=z_initial, a=y, edgeWeights=edge_weights, useConvex=1, epsilon=0.01, mu=0.5)
        print('Network_lasso_cvx:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)

    # print('obj_cvx', obj)
    # print(runtime)
    # print(objerror)
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
    plt.xlim((0, 120))
    plt.ylim((1e-6, 1e1))
    # plt.title('Network Lasso Algorithm (cvx)')
    plt.title(r'Network Lasso Algorithm ' r'(cvx, $\lambda=0.5$)', fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    # plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/GridGraph_cvx/Network_lamb=05_grid_graph_cvx', bbox_inches='tight')


if __name__ == "__main__":
    # obj_find()
    gfl_cvx_1()
    # gfl_cvx_2()
    network_lasso_cvx_1()
    # network_lasso_cvx_2()

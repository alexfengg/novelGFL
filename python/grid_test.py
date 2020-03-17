'''2D grid graph of an image, cvx algorithms comparision '''
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from snap import *
from solver import solve_GroupFL
from libGroupFL import coloredge, decompose_graph
from solver_pool import solve_GroupFL_pool
from NetworkLasso_solver_Hallac import runADMM
from NetworkLasso_solver import solve_NetworkLasso


def gfl_cvx_1():
    ''' group fused lasso algorithm with explicit updates '''
    img = np.random.normal(0, 1, (10, 10, 3))

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
    # rho_set = [2**k for k in range(7)]
    rho_set = [1]

    x_hat1, obj, _, _ = solve_GroupFL(
        y=data, G=graph, G0=graph0, G1=graph1, maxsteps=50, rho=1, lam=10)
    optobj = obj[-1]
    print("obj_exp:", obj)

    # lam = 10
    # objtemp = (np.linalg.norm(x_hat1 - data))**2
    #
    # for node1, node2 in graph.edges():
    #     objtemp = objtemp + lam * np.linalg.norm(x_hat1[node1] - x_hat1[node2])
    #
    # print(objtemp)

    obj = []

    for rho in rho_set:
        clock_time = time.time()
        x_hat2, tevolution, obj, _ = solve_GroupFL_pool(
            y=data, G=graph, G0=graph0, G1=graph1, maxsteps=50, rho=rho, lam=10)
        print('GraphLasso_cvx:',
              time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)
        # print('x_cvx:',x_hat)

        # lam = 10
        # objtemp = (np.linalg.norm(x_hat2 - data))**2
        #
        # for node1, node2 in graph.edges():
        #     objtemp = objtemp + lam * \
        #         np.linalg.norm(x_hat2[node1] - x_hat2[node2])
        #
        # print(objtemp)
        # print("x1-x2", x_hat1 - x_hat2)

    print('obj_cvx:', obj)


def gfl_cvx_2():
    '''   Graph Fused Lasso   '''
    numNode = 100
    data = np.zeros((numNode, 2))
    for k in range(11):
        data[k] = data[k] + np.array([1, 1])
        data[k + 11] = data[k + 11] + np.array([-1, 1])
        data[k + 22] = data[k + 22] + np.array([2, 2])
        data[k + 33] = data[k + 33] + np.array([-1, -1])

    np.random.seed(1)
    noise = 0.5 * np.random.normal(0, 1, (numNode, 2))
    data = data + noise

    graph = nx.grid_2d_graph(10, 10)
    graph = nx.convert_node_labels_to_integers(graph)

    print('graph:',list(graph.edges()))

    temp_edges = []
    for k in range(0, numNode):
        for j in graph.neighbors(k):
            if j == k + 1:
                temp_edges.append([k, j])

    graph0_edges = []
    graph0_edges.append(temp_edges[0])
    for edge in temp_edges:
        if (edge[0] != graph0_edges[-1][1]) and (edge!=graph0_edges[-1]):
            graph0_edges.append(edge)



    graph1_edges = [edge for edge in list(
        graph.edges()) if edge not in graph0_edges]

    graph1_edges.sort()

    graph0_nodes = np.unique(np.array(graph0_edges))
    graph1_nodes = np.unique(np.array(graph1_edges))

    graph0 = nx.Graph()
    graph1 = nx.Graph()
    graph0.add_nodes_from(graph0_nodes)
    graph1.add_nodes_from(graph1_nodes)
    graph0.add_edges_from(graph0_edges)
    graph1.add_edges_from(graph1_edges)

    print("graph0:",graph0_edges,"graph1:",graph1_edges)

    # edgelist = [[k, k + 1] for k in range(numNode - 1)]

    # graph = nx.Graph()
    # graph.add_edges_from(edgelist)
    # coloredge(graph)
    # graph0, graph1 = decompose_graph(graph)

    runtime = []
    objerror = []
    # rho_set = [2**k for k in range(7)]
    rho_set = [1]

    x_hat1, obj1, _, _ = solve_GroupFL(
        y=data, G=graph, G0=graph0, G1=graph1, maxsteps=20, rho=1, lam=1)
    optobj = obj1[-1]
    print(obj1[:20])

    for rho in rho_set:
        clock_time = time.time()
        x_hat2, tevolution, obj, _ = solve_GroupFL_pool(
            y=data, G=graph, G0=graph0, G1=graph1, maxsteps=20, rho=rho, lam=1)
        print('GraphLasso_pool:', time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)
        print(obj[:20])
        print((x_hat1 - x_hat2)[:10])
        print([obj1[i] - obj[i] for i in range(len(obj))])

    # plotting
    plt.switch_backend('TkAgg')
    plt.figure()
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    # plt.plot(runtime[1], objerror[1], label=r'$\rho=2^1$')
    # plt.plot(runtime[2], objerror[2], label=r'$\rho=2^2$')
    # plt.plot(runtime[3], objerror[3], label=r'$\rho=2^3$')
    # plt.plot(runtime[4], objerror[4], label=r'$\rho=2^4$')
    # plt.plot(runtime[5], objerror[5], label=r'$\rho=2^5$')
    # plt.plot(runtime[6], objerror[6], label=r'$\rho=2^6$')
    plt.legend()
    plt.xlim((0, 50))
    plt.ylim((1e-5, 1e2))
    plt.title('Graph Fused Lasso Algorithm (cvxpy)')
    plt.xlabel('Running time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    # plt.show()


if __name__ == "__main__":
    gfl_cvx_1()
    # gfl_cvx_2()

''' 1-D Chain Graph, using cvx packages '''
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from snap import *
from libGroupFL import coloredge, decompose_graph
from solver import solve_GroupFL
from solver_pool import solve_GroupFL_pool
from NetworkLasso_solver_Hallac import runADMM
from NetworkLasso_solver import solve_NetworkLasso


def solver_network_cvx_lambda_1():
    '''
    1-D Chain Graph, y[i]=y*[i]+epsilon, y*[0:11]=(1,1),y*[11:22]=(-1,1),
    y*[22:33]=(2,2),y*[33:44]=(-1,-1),y*[44:]=(0,0)
    lambda = 1
    '''
    # Set parameters
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

    y = np.transpose(data)
    p, n = y.shape

    print("The number of data is %d, the dimension of each data is %d" % (n, p))
    # Creating graph
    G = TUNGraph.New()
    for k in range(numNode):
        G.AddNode(k)

    for i in range(numNode - 1):
        G.AddEdge(i, i + 1)

    print('Number of Edges', G.GetEdges())

    edge_weights = TIntPrFltH()
    for edge in G.Edges():
        temp = TIntPr(edge.GetSrcNId(), edge.GetDstNId())
        edge_weights.AddDat(temp, 1)
        # print("edge (%d, %d)" % (edge.GetSrcNId(), edge.GetDstNId()))
        # weight = edgeWeights.GetDat(TIntPr(edge.GetSrcNId(), edge.GetDstNId()))
        # print("edgeWeights:",weight)

    nodes = G.GetNodes()
    # Initialize variables to 0
    x_initial = np.zeros((p, nodes))
    u_initial = np.zeros((p, 2 * G.GetEdges()))
    z_initial = np.zeros((p, 2 * G.GetEdges()))

    runtime = []
    objerror = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [1]

    '''
    find best obje value via explicit network lasso alogrithm
    '''

    graph = nx.Graph()
    graph.add_nodes_from(range(numNode))
    graph.add_edges_from([[k, k + 1] for k in range(numNode - 1)])

    _, obj, _, _ = solve_NetworkLasso(data, graph, maxsteps=4000, rho=4, lam=1)
    optobj = obj[-1]
    # print(x1)

    for rho in rho_set:
        clock_time = time.time()
        _, tevolution, obj, _ = runADMM(G1=G, sizeOptVar=p, sizeData=p, lamb=1, rho=rho, numiters=200, x=x_initial,
                                        u=u_initial, z=z_initial, a=y, edgeWeights=edge_weights, useConvex=1, epsilon=0.01, mu=0.5)
        print('NetworkLasso:', time.time() - clock_time, ',rho:', rho)

        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)
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
    plt.xlim((0, 50))
    plt.ylim((1e-5, 1e2))
    # plt.title('Network Lasso Algorithm-explict (lambda=1)')
    plt.title(r'Network Lasso Algorithm ' r'(cvx, $\lambda=1$)',fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/ChainGraph_cvx/Network_lamb=1_chain_graph_cvx', bbox_inches='tight')


def solver_network_cvx_lambda_2():
    '''
    1-D Chain Graph, y[i]=y*[i]+epsilon, y*[0:11]=(1,1),y*[11:22]=(-1,1),
    y*[22:33]=(2,2),y*[33:44]=(-1,-1),y*[44:]=(0,0)
    lambda = 1
    '''
    # Set parameters
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

    y = np.transpose(data)
    p, n = y.shape

    print("The number of data is %d, the dimension of each data is %d" % (n, p))
    # Creating graph
    G = TUNGraph.New()
    for k in range(numNode):
        G.AddNode(k)

    for i in range(numNode - 1):
        G.AddEdge(i, i + 1)

    print('Number of Edges', G.GetEdges())

    edge_weights = TIntPrFltH()
    for edge in G.Edges():
        temp = TIntPr(edge.GetSrcNId(), edge.GetDstNId())
        edge_weights.AddDat(temp, 1)
        # print("edge (%d, %d)" % (edge.GetSrcNId(), edge.GetDstNId()))
        # weight = edgeWeights.GetDat(TIntPr(edge.GetSrcNId(), edge.GetDstNId()))
        # print("edgeWeights:",weight)

    nodes = G.GetNodes()
    # Initialize variables to 0
    x_initial = np.zeros((p, nodes))
    u_initial = np.zeros((p, 2 * G.GetEdges()))
    z_initial = np.zeros((p, 2 * G.GetEdges()))

    runtime = []
    objerror = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [4]

    '''
    find best obje value via explicit network lasso alogrithm
    '''

    graph = nx.Graph()
    graph.add_nodes_from(range(numNode))
    graph.add_edges_from([[k, k + 1] for k in range(numNode - 1)])

    _, obj, _, _ = solve_NetworkLasso(
        data, graph, maxsteps=4000, rho=4, lam=0.1)
    optobj = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        _, tevolution, obj, _ = runADMM(G1=G, sizeOptVar=p, sizeData=p, lamb=0.1, rho=rho, numiters=200, x=x_initial,
                                        u=u_initial, z=z_initial, a=y, edgeWeights=edge_weights, useConvex=1, epsilon=0.01, mu=0.5)
        print('NetworkLasso:', time.time() - clock_time, ',rho:', rho)

        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)
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
    plt.xlim((0, 50))
    plt.ylim((1e-5, 1e2))
    # plt.title('Network Lasso Algorithm-cvxpy (lambda=0.1)')
    plt.title(r'Network Lasso Algorithm ' r'(cvx, $\lambda=0.1$)',fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig('Figures/ChainGraph_cvx/Network_lamb=01_chain_graph_cvx',
                bbox_inches='tight')


def gfl_cvx_1():
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

    edgelist = [[k, k + 1] for k in range(numNode - 1)]

    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    coloredge(graph)
    graph0, graph1 = decompose_graph(graph)

    runtime = []
    objerror = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [1]

    _, obj, _, _ = solve_GroupFL(
        y=data, G=graph, G0=graph0, G1=graph1, maxsteps=3000, rho=8, lam=1)
    optobj = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        _, tevolution, obj, _ = solve_GroupFL_pool(
            y=data, G=graph, G0=graph0, G1=graph1, maxsteps=250, rho=rho, lam=1)
        print('GraphLasso_pool:', time.time() - clock_time, ',rho:', rho)
        objerror.append([i - optobj for i in obj])
        runtime.append(tevolution)

    # plotting
    plt.figure(dpi=100)
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.plot(runtime[1], objerror[1], label=r'$\rho=2^1$')
    plt.plot(runtime[2], objerror[2], label=r'$\rho=2^2$')
    plt.plot(runtime[3], objerror[3], label=r'$\rho=2^3$')
    plt.plot(runtime[4], objerror[4], label=r'$\rho=2^4$')
    plt.plot(runtime[5], objerror[5], label=r'$\rho=2^5$')
    plt.plot(runtime[6], objerror[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2))
    plt.ylim((1e-12, 1e2))
    # plt.title('Graph Fused Lasso Algorithm-cvxpy(lambda=1)')
    plt.title(r'Graph Fused Lasso Algorithm ' r'(cvx, $\lambda=1$)',fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig('Figures/ChainGraph_cvx/GraphFL_lamb=1_chain_graph_cvx',
                bbox_inches='tight')


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

    edgelist = [[k, k + 1] for k in range(numNode - 1)]

    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    coloredge(graph)
    graph0, graph1 = decompose_graph(graph)

    runtime = []
    objerror = []
    rho_set = [2**k for k in range(7)]
    # rho_set = [1]

    _, obj, _, _ = solve_GroupFL(
        y=data, G=graph, G0=graph0, G1=graph1, maxsteps=3000, rho=8, lam=0.1)
    optobj = obj[-1]

    for rho in rho_set:
        clock_time = time.time()
        _, tevolution, obj, _ = solve_GroupFL_pool(
            y=data, G=graph, G0=graph0, G1=graph1, maxsteps=250, rho=rho, lam=0.1)
        print('GraphLasso_pool:', time.time() - clock_time, ',rho:', rho)
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
    plt.xlim((0, 50))
    plt.ylim((1e-5, 1e2))
    # plt.title('Graph Fused Lasso Algorithm-cvxpy (lambda=0.1)')
    plt.title(r'Graph Fused Lasso Algorithm ' r'(cvx, $\lambda=0.1$)',fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig('Figures/ChainGraph_cvx/GraphFL_lamb=01_chain_graph_cvx',
                bbox_inches='tight')


if __name__ == "__main__":
    # solver_network_cvx_lambda_1()
    # solver_network_cvx_lambda_2()
    gfl_cvx_1()
    # gfl_cvx_2()

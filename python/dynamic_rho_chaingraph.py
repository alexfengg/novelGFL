import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from solver import *
from NetworkLasso_solver import solve_NetworkLasso

# from libGroupFL import coloredge, decompose_graph
# from solver import solve_GroupFL
# from solver_pool import solve_GroupFL_pool
# from NetworkLasso_solver_Hallac import runADMM
# # from NetworkLasso_solver import solve_NetworkLasso


def solver_gfl_explicit_lambda_1_dr():
    numNode = 100
    numRho = 7
    data = np.zeros((numNode, 2))
    for k in range(11):
        data[k] = data[k] + np.array([1, 1])
        data[k + 11] = data[k + 11] + np.array([-1, 1])
        data[k + 22] = data[k + 22] + np.array([2, 2])
        data[k + 33] = data[k + 33] + np.array([-1, -1])

    np.random.seed(1)
    noise = 0.5 * np.random.normal(0, 1, (numNode, 2))
    data = data + noise
    # print('sum of data:',np.sum(data))

    edgelist = [[k, k + 1] for k in range(numNode - 1)]

    G = nx.Graph()
    G.add_nodes_from(range(numNode))
    G.add_edges_from(edgelist)
    coloredge(G)
    G0, G1 = decompose_graph(G)

    '''
    lam=1
    '''

    objerror = []
    runtime = []

    _, obj, _, _ = solve_GroupFL(data, G, G0, G1, maxsteps=5000, rho=1, lam=1)
    optobj1 = obj[-1]

    obj = []
    t = time.time()
    x, obj, err, tevolution = solve_GroupFL(
        data, G, G0, G1, maxsteps=700, rho=1, lam=1, verbose=1)
    print('running time GraphLasso_explicit:',
          time.time() - t, ',rho: dynamic')
    objerror.append([i - optobj1 for i in obj])
    runtime.append(tevolution)

    # plotting
    # plt.figure(figsize=(9,6),dpi=200)
    plt.figure(dpi=100)
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2))
    # plt.xlim((0,400))
    plt.ylim((1e-12, 1e2))
    # plt.title('Graph Fused Lasso Algorithm (explict & lambda=1)')
    plt.title(
        r'Graph Fused Lasso Algorithm ' r'(explicit, $\lambda=1$)', fontsize=12)
    plt.xlabel('Running Time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/ChainGraph/dynamica_rho_GFL_lamb=1_chain_graph_explict', bbox_inches='tight')


def solver_gfl_explicit_lambda_01_dr():
    numNode = 100
    numRho = 7
    data = np.zeros((numNode, 2))
    for k in range(11):
        data[k] = data[k] + np.array([1, 1])
        data[k + 11] = data[k + 11] + np.array([-1, 1])
        data[k + 22] = data[k + 22] + np.array([2, 2])
        data[k + 33] = data[k + 33] + np.array([-1, -1])

    np.random.seed(1)
    noise = 0.5 * np.random.normal(0, 1, (numNode, 2))
    data = data + noise
    # print('sum of data:',np.sum(data))

    edgelist = [[k, k + 1] for k in range(numNode - 1)]

    G = nx.Graph()
    G.add_nodes_from(range(numNode))
    G.add_edges_from(edgelist)
    coloredge(G)
    G0, G1 = decompose_graph(G)

    '''
    lam=0.1
    '''

    objerror = []
    runtime = []

    _, obj, _, _ = solve_GroupFL(
        data, G, G0, G1, maxsteps=5000, rho=1, lam=0.1)
    optobj1 = obj[-1]

    obj = []
    t = time.time()
    x, obj, err, tevolution = solve_GroupFL(
        data, G, G0, G1, maxsteps=700, rho=1, lam=0.1, verbose=1)
    print('running time GraphLasso_explicit:',
          time.time() - t, ',rho: dynamic')
    objerror.append([i - optobj1 for i in obj])
    runtime.append(tevolution)

    # plotting
    # plt.figure(figsize=(9,6),dpi=200)
    plt.figure(dpi=100)
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2))
    # plt.xlim((0,400))
    plt.ylim((1e-12, 1e2))
    # plt.title('Graph Fused Lasso Algorithm (explict & lambda=1)')
    plt.title(
        r'Graph Fused Lasso Algorithm ' r'(explicit, $\lambda=0.1$)', fontsize=12)
    plt.xlabel('Running Time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/ChainGraph/dynamica_rho_GFL_lamb=01_chain_graph_explict', bbox_inches='tight')


def solver_nfl_explicit_lambda_1_dr():
    '''
    Network Lasso explict
    lam=1
    '''

    numNode = 100
    numRho = 7
    data = np.zeros((numNode, 2))
    for k in range(11):
        data[k] = data[k] + np.array([1, 1])
        data[k + 11] = data[k + 11] + np.array([-1, 1])
        data[k + 22] = data[k + 22] + np.array([2, 2])
        data[k + 33] = data[k + 33] + np.array([-1, -1])

    np.random.seed(1)
    noise = 0.5 * np.random.normal(0, 1, (numNode, 2))
    data = data + noise
    # print('sum of data:',np.sum(data))

    edgelist = [[k, k + 1] for k in range(numNode - 1)]

    G = nx.Graph()
    G.add_nodes_from(range(numNode))
    G.add_edges_from(edgelist)
    coloredge(G)
    G0, G1 = decompose_graph(G)

    objerror = []
    runtime = []
    _, obj, _, _ = solve_NetworkLasso(data, G, maxsteps=10000, rho=1, lam=1)
    optobj3 = obj[-1]

    obj = []
    t = time.time()
    x, obj, err, tevolution = solve_NetworkLasso(
        data, G, maxsteps=700, rho=2, lam=1, verbose=1)
    print('running time Network Lasso explict:',
          time.time() - t, ',rho: dynamic')
    objerror.append([i - optobj3 for i in obj])
    runtime.append(tevolution)

    # plotting
    plt.figure()
    plt.plot(runtime[0], objerror[0], label=r'$\rho=2^0$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2))
    # plt.xlim((0,400))
    plt.ylim((1e-12, 1e2))
    plt.title(r'Network Lasso Algorithm ' r'(explicit, $\lambda=1$)', fontsize=12)
    plt.xlabel('Running time (seconds)', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig('Figures/ChainGraph/dynamic_rho_Network_lamb=1_chain_graph_explicit',
                bbox_inches='tight')


if __name__ == "__main__":
    # solver_gfl_explicit_lambda_1_dr()
    solver_gfl_explicit_lambda_01_dr()
    # solver_nfl_explicit_lambda_1_dr()

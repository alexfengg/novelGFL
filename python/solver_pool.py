import sys
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import NGroupFL
import multiprocessing as mp
import cvxpy as cp
from libGroupFL import *


def poolfst(data):
    node1, node2 = data[0]
    tempx = data[1]
    zeros = data[2]
    G1 = data[3]
    u = data[4]
    z = data[5]
    rho = data[6]
    lam = data[7]
    y = data[8]
    nodelist = data[9]

    n, p = y.shape
    tnode1 = zeros
    tnode2 = zeros
    ds = 0
    dt = 0

    if node1 in G1.nodes():
        ds = len(list(G1.neighbors(node1)))
        for k in nodelist[node1].neighbor_right_idx:
            tnode1 = tnode1 - u[k] - rho * z[k]

        for k in nodelist[node1].neighbor_left_idx:
            tnode1 = tnode1 + u[k] + rho * z[k]

        for k in G1.neighbors(node1):
            tnode1 = tnode1 - rho * tempx[node1] - rho * tempx[k]

    if node2 in G1.nodes():
        dt = len(list(G1.neighbors(node2)))
        for k in nodelist[node2].neighbor_right_idx:
            tnode2 = tnode2 - u[k] - rho * z[k]

        for k in nodelist[node2].neighbor_left_idx:
            tnode2 = tnode2 + u[k] + rho * z[k]

        for k in G1.neighbors(node2):
            tnode2 = tnode2 - rho * tempx[node2] - rho * tempx[k]

    # print('node:',node1,'tnode:',tnode1)

    xs = cp.Variable(p)
    xt = cp.Variable(p)

    fs = cp.sum_squares(xs - y[node1])
    ft = cp.sum_squares(xt - y[node2])
    g = lam * cp.norm(xs - xt) + rho * ds * cp.sum_squares(xs) + rho * dt * cp.sum_squares(xt)
    h = 0
    for k in range(p):
        h = h + tnode1[0][k] * xs[k] + tnode2[0][k] * xt[k]
    objective = cp.Minimize(fs + ft + g + h)
    problem = cp.Problem(objective)
    problem.solve()

    return xs.value, xt.value


def poolf(data):
    node = data[0]
    tempx = data[1]
    zeros = data[2]
    G1 = data[3]
    u = data[4]
    z = data[5]
    rho = data[6]
    lam = data[7]
    y = data[8]
    nodelist = data[9]

    n, p = y.shape
    tnode = zeros
    deg = len(list(G1.neighbors(node)))

    for k in nodelist[node].neighbor_right_idx:
        tnode = tnode - u[k] - rho * z[k]

    for k in nodelist[node].neighbor_left_idx:
        tnode = tnode + u[k] + rho * z[k]

    for k in G1.neighbors(node):
        tnode = tnode - rho * tempx[node] - rho * tempx[k]

    # cvx
    xnew = cp.Variable(p)
    f = cp.sum_squares(xnew - y[node])
    g = rho * deg * cp.sum_squares(xnew)
    h = 0
    for k in range(p):
        h = h + tnode[0][k] * xnew[k]
    objective = cp.Minimize(f + g + h)
    problem = cp.Problem(objective)
    problem.solve()

    return xnew.value

    # explicit solution
    # xnew=(2*y[node]-tnode)/2/(1+rho*deg)
    # return xnew


def solve_GroupFL_pool(y, G, G0, G1, maxsteps=100, rho=1, lam=0.5, verbose=0):
    t = 0
    # n: number of all nodes, p: dimension of data point
    n, p = y.shape
    # print('y',y)

    # define a zero vector with dimension p
    zeros = np.zeros((1, p))

    # initial x, a n*p zero array
    x = np.zeros(y.shape)

    # initial u,z, u[i], z[i] are i-th edge of G1 ,initialized a 1*p array
    u = np.zeros((len(G1.edges()), p))
    z = np.zeros((len(G1.edges()), p))

    # list of all edges of G1
    G1edge = list(G1.edges())
    G0edge = list(G0.edges())
    G1edge.sort()

    # matrix A for Ax+Bz=C, for stopping criterion & dynamic rho
    if verbose:
        A=np.zeros((len(G1edge),n))
        for k in range(len(G1edge)):
            A[k,G1edge[k][0]]=1
            A[k,G1edge[k][1]]=-1
        r=1
        s=1

    nodelist = []
    for node in G.nodes():
        nodelist.append(NGroupFL.Node(node, G1))

    # r=1
    # s=1

    pool = mp.Pool(processes=mp.cpu_count() - 1)

    dt = []
    obj = []
    objerr = []
    # while (t<=maxsteps) and (r>1e-10 or s>1e-10):
    while (t <= maxsteps):
        sys.stdout.write('\r'+'gfl_cvx_status:'+str(int(100*t/maxsteps))+'%')
        t += 1

        t0 = time.time()
        tempx = x+0.0000
        # print(tempx[0])

        # varying rho if verbose = 1
        if verbose:
            if r>10*s:
                rho = 2*rho
            elif s>10*r:
                rho = rho/2

        # update x for nodes in E0
        data = [[G0edge[k], tempx, zeros, G1, u, z, rho, lam, y, nodelist]
                for k in range(len(G0edge))]

        temp = pool.map(poolfst, data)

        for k in range(len(G0edge)):
            x[G0edge[k][0]] = temp[k][0]
            x[G0edge[k][1]] = temp[k][1]

        # update x for nodes in E1/E0
        # data=[[node,tempx,zeros,G1,u,z,rho,lam,y,nodelist] for node in G1.nodes() if node not in G0.nodes()]
        data = []
        for node in G1.nodes():
            if node not in G0.nodes():
                data.append([node, tempx, zeros, G1, u,
                             z, rho, lam, y, nodelist])

        temp = pool.map(poolf, data)
        counter = 0
        for node in G1.nodes():
            if node not in G0.nodes():
                x[node] = temp[counter]
                counter += 1

        # print('x-cvx:',x)
        # update for z
        tempz = z + 0.000
        for k in range(len(G1edge)):
            z[k] = threshold(x[G1edge[k][0]] - x[G1edge[k][1]] - u[k] / rho, lam / rho)

        # update for u
        for k in range(len(G1edge)):
            u[k] = u[k] + rho * (z[k] - x[G1edge[k][0]] + x[G1edge[k][1]])

        # stopping criterion
        if verbose:
            s_matrix=rho*np.dot(np.transpose(A), z-tempz)
            r_matrix=np.dot(A,x)-z
            s=np.linalg.norm(s_matrix)
            r=np.linalg.norm(r_matrix)

        # record runtime for each update
        t1 = time.time() - t0
        dt.append(t1)

        # objtemp = (LA.norm(x-a))**2+LA.norm(x)
        objtemp = (np.linalg.norm(x - y))**2

        for node1, node2 in G.edges():
            objtemp = objtemp + lam * np.linalg.norm(x[node1] - x[node2])

        obj.append(objtemp)

    pool.close()
    pool.join()


    for k in range(len(obj)):
        objerr.append(obj[k] - obj[-1])

    tevolution = []
    temp = 0

    for k in range(t):
        temp = temp + dt[k]
        tevolution.append(temp)

    return x, tevolution, obj, objerr

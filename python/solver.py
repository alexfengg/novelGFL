'''
Solver for Group Fused Lasso Algorithm, requires networkx
'''
import sys
import time
import numpy as np
import NGroupFL
from libGroupFL import *


def solve_GroupFL(y, G, G0, G1, maxsteps=100, rho=1, lam=0.5, verbose=0):
    ''' The solver needs packages networkx '''
    t = 0
    # n: number of all nodes, p: dimension of data point
    n, p = y.shape
    zeros = np.zeros((1, p))
    x = np.zeros(y.shape)


    obj = []
    objerr = []
    dt = []

    # initial u,z, u[i], z[i] are i-th edge of G1 ,initialized a 1*p array
    u = np.zeros((len(G1.edges()), p))
    z = np.zeros((len(G1.edges()), p))

    # list of all edges of G1
    G1edge = list(G1.edges())

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

    while (t <= maxsteps):
        # while (t<=maxsteps) and (r>1e-10 or s>1e-10):
        sys.stdout.write('\r'+'gfl_explicit_status:'+str(int(100*t/maxsteps))+'%')
        t += 1
        # print('time:',t,'x:',x)

        # varying rho if verbose = 1
        if verbose:
            if r>10*s:
                rho = 2*rho
            elif s>10*r:
                rho = rho/2

        # in python, tempx = x returns a variable 'tempx' which is only a refence of x, a change of x will automatically result in a change of tempx
        tempx = x + 0.0000
        # print(tempx[0])
        t0 = time.time()
        # update x for nodes in E0
        for node1, node2 in G0.edges():
            tnode1 = zeros
            tnode2 = zeros
            c1 = 1
            c2 = 1

            if node1 in G1.nodes():
                c1 = 1 + rho * (len(list(G1.neighbors(node1))))
                for k in nodelist[node1].neighbor_right_idx:
                    tnode1 = tnode1 - u[k] - rho * z[k]

                for k in nodelist[node1].neighbor_left_idx:
                    tnode1 = tnode1 + u[k] + rho * z[k]

                for k in G1.neighbors(node1):
                    tnode1 = tnode1 - rho * tempx[node1] - rho * tempx[k]

            if node2 in G1.nodes():
                c2 = 1 + rho * (len(list(G1.neighbors(node2))))
                for k in nodelist[node2].neighbor_right_idx:
                    tnode2 = tnode2 - u[k] - rho * z[k]

                for k in nodelist[node2].neighbor_left_idx:
                    tnode2 = tnode2 + u[k] + rho * z[k]

                for k in G1.neighbors(node2):
                    tnode2 = tnode2 - rho * tempx[node2] - rho * tempx[k]

            # print('c1,c2:',c1,c2)
            # print('tnode1,tnode2:',tnode1,tnode2)
            # print('explicit,node:',node1,'tnode:',tnode1)
            x[node1], x[node2] = solve_f(
                (2 * y[node1] - tnode1) / 2 / c1, (2 * y[node2] - tnode2) / 2 / c2, lam, c1, c2)

        # update x for nodes in E1/E0
        for node in G1.nodes():
            if node not in G0.nodes():
                tnode = zeros
                c = 1 + rho * (len(list(G1.neighbors(node))))

                for k in nodelist[node].neighbor_right_idx:
                    tnode = tnode - u[k] - rho * z[k]

                for k in nodelist[node].neighbor_left_idx:
                    tnode = tnode + u[k] + rho * z[k]

                for k in G1.neighbors(node):
                    tnode = tnode - rho * tempx[node] - rho * tempx[k]

                x[node] = (2 * y[node] - tnode) / 2 / c

        # update for z
        tempz = z+0.0000
        for k in range(len(G1edge)):
            z[k] = threshold(x[G1edge[k][0]] - x[G1edge[k]
                                                 [1]] - u[k] / rho, lam / rho)

        # update for u
        for k in range(len(G1edge)):
            u[k] = u[k] + rho * (z[k] - x[G1edge[k][0]] + x[G1edge[k][1]])

        # record runtime for each update
        t1 = time.time() - t0
        dt.append(t1)

        # stopping criterion
        if verbose:
            s_matrix=rho*np.dot(np.transpose(A), z-tempz)
            r_matrix=np.dot(A,x)-z
            s=np.linalg.norm(s_matrix)
            r=np.linalg.norm(r_matrix)

        # print('s,r:',s,r)

        objtemp = (np.linalg.norm(x - y))**2

        for node1, node2 in G.edges():
            objtemp = objtemp + lam * np.linalg.norm(x[node1] - x[node2])

        obj.append(objtemp)

    tevolution = []
    temp = 0

    for k in range(len(dt)):
        temp = temp + dt[k]
        tevolution.append(temp)

    for k in range(len(obj)):
        objerr.append(obj[k] - obj[-1])

    return x, obj, objerr, tevolution

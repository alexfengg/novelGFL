import csv
import numpy as np
import matplotlib.pyplot as plt

# from libGroupFL import coloredge, decompose_graph
# from solver import solve_GroupFL
# from solver_pool import solve_GroupFL_pool
# from NetworkLasso_solver_Hallac import runADMM
# # from NetworkLasso_solver import solve_NetworkLasso


def plot_gfl_005():
    obj005 = []
    with open('Data/ChiCrime005_algorithm1_obj_rhos.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for value in spamreader:
            obj005.append(value[-1])

    obj5 = [2.22448786]

    for k in range(1, len(obj005)):
        obj5.append(float(obj005[k]))
    #
    # print(len(obj5))
    obj_005 = [k - 0.641061419585920 for k in obj5]

    # obj_005= [float(k)-0.64106 for k in obj005]
    obj1 = [obj_005[2000 * k:2000 * (k + 1)] for k in range(7)]

    plt.figure()
    plt.plot(obj1[0], label=r'$\rho=2^0$')
    plt.plot(obj1[1], label=r'$\rho=2^1$')
    plt.plot(obj1[2], label=r'$\rho=2^2$')
    plt.plot(obj1[3], label=r'$\rho=2^3$')
    plt.plot(obj1[4], label=r'$\rho=2^4$')
    plt.plot(obj1[5], label=r'$\rho=2^5$')
    plt.plot(obj1[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2000))
    plt.ylim((1e-8, 1e1))
    plt.title(r'Graph Fused Lasso Algorithm ' r'($\lambda=0.05$)',fontsize=12)
    # plt.xlabel('Running time (seconds)', fontsize=12)
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/Chicago/GraphFL_CHI_lamb=005', bbox_inches='tight')
    # plt.show()


def plot_gfl_025():
    obj005 = []
    with open('Data/ChiCrime025_algorithm1_obj_rhos.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for value in spamreader:
            obj005.append(value[-1])

    obj5 = [4.27569477]

    for k in range(1, len(obj005)):
        obj5.append(float(obj005[k]))
    #
    # print(len(obj5))
    obj_005 = [k - 0.837281918719618 for k in obj5]

    # obj_005= [float(k)-0.64106 for k in obj005]
    obj1 = [obj_005[2000 * k:2000 * (k + 1)] for k in range(7)]

    plt.figure()
    plt.plot(obj1[0], label=r'$\rho=2^0$')
    plt.plot(obj1[1], label=r'$\rho=2^1$')
    plt.plot(obj1[2], label=r'$\rho=2^2$')
    plt.plot(obj1[3], label=r'$\rho=2^3$')
    plt.plot(obj1[4], label=r'$\rho=2^4$')
    plt.plot(obj1[5], label=r'$\rho=2^5$')
    plt.plot(obj1[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2000))
    plt.ylim((1e-8, 1e1))
    plt.title(r'Graph Fused Lasso Algorithm ' r'($\lambda=0.25$)',fontsize=12)
    # plt.xlabel('Running time (seconds)', fontsize=12)
    plt.xlabel('Number of Iterations',  fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/Chicago/GraphFL_CHI_lamb=025', bbox_inches='tight')
    # plt.show()


def plot_netlasso_005():
    obj005 = []
    with open('Data/ChiCrime005_Network_obj_rhos.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for value in spamreader:
            obj005.append(value[-1])

    obj5 = [2.46125401]

    for k in range(1, len(obj005)):
        obj5.append(float(obj005[k]))
    #
    # print(len(obj5))
    obj_005 = [k - 0.641061419585920 for k in obj5]

    # obj_005= [float(k)-0.64106 for k in obj005]
    obj1 = [obj_005[2000 * k:2000 * (k + 1)] for k in range(7)]

    plt.figure()
    plt.plot(obj1[0], label=r'$\rho=2^0$')
    plt.plot(obj1[1], label=r'$\rho=2^1$')
    plt.plot(obj1[2], label=r'$\rho=2^2$')
    plt.plot(obj1[3], label=r'$\rho=2^3$')
    plt.plot(obj1[4], label=r'$\rho=2^4$')
    plt.plot(obj1[5], label=r'$\rho=2^5$')
    plt.plot(obj1[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2000))
    plt.ylim((1e-8, 1e1))
    plt.title(r'Network Lasso Algorithm ' r'($\lambda=0.05$)',fontsize=12)
    # plt.xlabel('Running time (seconds)', fontsize=12)
    plt.xlabel('Number of Iterations',  fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/Chicago/NetworkLasso_CHI_lamb=005', bbox_inches='tight')
    # plt.show()


def plot_netlasso_025():
    obj005 = []
    with open('Data/ChiCrime025_Network_obj_rhos.csv', newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for value in spamreader:
            obj005.append(value[-1])

    obj5 = [6.90915168]

    for k in range(1, len(obj005)):
        obj5.append(float(obj005[k]))
    #
    # print(len(obj5))
    obj_005 = [k - 0.837281918719618 for k in obj5]

    # obj_005= [float(k)-0.64106 for k in obj005]
    obj1 = [obj_005[2000 * k:2000 * (k + 1)] for k in range(7)]

    plt.figure()
    plt.plot(obj1[0], label=r'$\rho=2^0$')
    plt.plot(obj1[1], label=r'$\rho=2^1$')
    plt.plot(obj1[2], label=r'$\rho=2^2$')
    plt.plot(obj1[3], label=r'$\rho=2^3$')
    plt.plot(obj1[4], label=r'$\rho=2^4$')
    plt.plot(obj1[5], label=r'$\rho=2^5$')
    plt.plot(obj1[6], label=r'$\rho=2^6$')
    plt.legend(loc='upper right')
    plt.xlim((0, 2000))
    plt.ylim((1e-8, 1e1))
    plt.title(r'Network Lasso Algorithm ' r'($\lambda=0.25$)',fontsize=12)
    # plt.xlabel('Running time (seconds)', fontsize=12)
    plt.xlabel('Number of Iterations',  fontsize=12)
    plt.ylabel('Objective Value Error', fontsize=12)
    plt.yscale('log')
    plt.savefig(
        'Figures/Chicago/NetworkLasso_CHI_lamb=025', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    plot_gfl_005()
    plot_gfl_025()
    plot_netlasso_005()
    plot_netlasso_025()

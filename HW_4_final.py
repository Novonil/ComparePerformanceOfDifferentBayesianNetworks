import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
import argparse
import warnings
start = time.time()
warnings.filterwarnings("error")


def chow_liu_algo( train_matrix, vmat, tsmat):
    train_valid_matrix = np.concatenate(( train_matrix, vmat), axis=0)
    pos_prob =  train_valid_matrix.sum(axis=0)
    pos_prob = pos_prob + 1
    neg_prob =  train_valid_matrix.shape[0] - pos_prob + 2
    pos_prob = np.true_divide(pos_prob,  train_valid_matrix.shape[0] + 2)
    neg_prob = np.true_divide(neg_prob,  train_valid_matrix.shape[0] + 2)
    Weight = np.zeros(( train_valid_matrix.shape[1],  train_valid_matrix.shape[1]))
    for col1 in range(int( train_valid_matrix.shape[1])):
        for col2 in range(int( train_valid_matrix.shape[1])):
            if col1 == col2:
                pass
            else:
                a00 = ( train_valid_matrix[:, col1] == 0) & ( train_valid_matrix[:, col2] == 0)
                a01 = ( train_valid_matrix[:, col1] == 0) & ( train_valid_matrix[:, col2] == 1)
                a10 = ( train_valid_matrix[:, col1] == 1) & ( train_valid_matrix[:, col2] == 0)
                a11 = ( train_valid_matrix[:, col1] == 1) & ( train_valid_matrix[:, col2] == 1)
                b00 = a00.sum()
                b01 = a01.sum()
                b10 = a10.sum()
                b11 = a11.sum()
                #add laplace smoothing if things dont go well
                P00 = np.true_divide(b00 + 1, int( train_valid_matrix.shape[0]) + 4)
                P01 = np.true_divide(b01 + 1, int( train_valid_matrix.shape[0]) + 4)
                P10 = np.true_divide(b10 + 1, int( train_valid_matrix.shape[0]) + 4)
                P11 = np.true_divide(b11 + 1, int( train_valid_matrix.shape[0]) + 4)
                W = P00 * np.log(np.true_divide(P00, (neg_prob[col1] * neg_prob[col2])))
                W = W + P01 * np.log(np.true_divide(P01, (neg_prob[col1] * pos_prob[col2])))
                W = W + P10 * np.log(np.true_divide(P10, (pos_prob[col1] * neg_prob[col2])))
                W = W + P11 * np.log(np.true_divide(P11, (pos_prob[col1] * pos_prob[col2])))
                Weight[col1][col2] = W
    Weight = Weight * (-1)
    e = csr_matrix(Weight)
    f = minimum_spanning_tree(e).toarray()
    f_csr = csr_matrix(f)
    l1, l2 = f_csr.toarray().nonzero()
    edges = zip(l1, l2)
    graph = Graph()
    for e in edges:
        graph.addEdge(e[0], e[1])
#The 0th feature is chosen as the root node
    graph.DFS(0, Weight.shape[0])
    o = graph.order
    pa = {o[0]: np.nan}
    for i in range(1, len(o)):
        if o[i] in graph.graph[o[i - 1]]:
            pa[o[i]] = o[i - 1]
        else:
            for j in range(i - 1):
                if o[i] in graph.graph[o[i - j - 2]]:
                    pa[o[i]] = o[i - j - 2]
                    break
                else:
                    pass

    cpt_matrix = []
    for child in list(pa.keys())[1:]:
        A00 = ( train_valid_matrix[:, child] == 0) & ( train_valid_matrix[:, pa[child]] == 0)
        A01 = ( train_valid_matrix[:, child] == 1) & ( train_valid_matrix[:, pa[child]] == 0)
        A10 = ( train_valid_matrix[:, child] == 0) & ( train_valid_matrix[:, pa[child]] == 1)
        A11 = ( train_valid_matrix[:, child] == 1) & ( train_valid_matrix[:, pa[child]] == 1)
        B00 = A00.sum()
        B01 = A01.sum()
        B10 = A10.sum()
        B11 = A11.sum()
        # temp1 = ( train_valid_matrix[:, pa[child]] == 0).sum()
        # temp2 = ( train_valid_matrix[:, pa[child]] == 1).sum()
        p00 = np.true_divide(B00 + 1, ( train_valid_matrix[:, pa[child]] == 0).sum() + 2)
        p01 = np.true_divide(B01 + 1, ( train_valid_matrix[:, pa[child]] == 0).sum() + 2)
        p10 = np.true_divide(B10 + 1, ( train_valid_matrix[:, pa[child]] == 1).sum() + 2)
        p11 = np.true_divide(B11 + 1, ( train_valid_matrix[:, pa[child]] == 1).sum() + 2)
        cpt_matrix.append([p00, p01, p10, p11])
    cpt_matrix = np.array(cpt_matrix)
    lcpt_matrix = np.log(cpt_matrix)
    pos_prob_X0 = np.log(np.true_divide(( train_valid_matrix[:, 0].sum() + 1),  train_valid_matrix.shape[0] + 2))
    neg_prob_X0 = np.log(np.true_divide(( train_valid_matrix.shape[0] -  train_valid_matrix[:, 0].sum() + 1),  train_valid_matrix.shape[0] + 2))
    t = tsmat.copy()
    t = t.astype(float)
    for i in range(1,  train_valid_matrix.shape[1]):
        par = pa[i]
        t[(tsmat[:, i] == 0) & (tsmat[:, par] == 0), i] = lcpt_matrix[i - 1][0]
        t[(tsmat[:, i] == 1) & (tsmat[:, par] == 0), i] = lcpt_matrix[i - 1][1]
        t[(tsmat[:, i] == 0) & (tsmat[:, par] == 1), i] = lcpt_matrix[i - 1][2]
        t[(tsmat[:, i] == 1) & (tsmat[:, par] == 1), i] = lcpt_matrix[i - 1][3]

    t[(tsmat[:, 0]) == 0, 0] = neg_prob_X0
    t[(tsmat[:, 0]) == 1, 0] = pos_prob_X0

    LLH_col = t.sum(axis=1)
    mLL = LLH_col.mean()
    return mLL


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.order = []
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def DFS(self, v, vertex):
        visited = [False]*vertex
        self.DFSUtil(v, visited)

    def DFSUtil(self,v,visited):
        visited[v]=True
        self.order.append(v)

        for i in self.graph[v]:
            if visited[i] == False:
                # print(visited)
                self.DFSUtil(i,visited)



def IBN_algo( train_matrix, vmat, tsmat):
    train_valid_matrix = np.concatenate(( train_matrix, vmat), axis = 0)
    pos_prob = train_valid_matrix.sum(axis=0)
    neg_prob =  train_valid_matrix.shape[0] - pos_prob
    pos_prob = pos_prob + 1
    neg_prob = neg_prob + 1
    lpos_prob = np.log(np.true_divide(pos_prob,  train_valid_matrix.shape[0] + 2))
    lneg_prob = np.log(np.true_divide(neg_prob,  train_valid_matrix.shape[0] + 2))
    tes1 = (tsmat * lpos_prob).astype(float)
    tes2 = np.ma.array(tes1, mask=tes1 != 0)
    tes3 = tes2 + lneg_prob
    tes4 = tes3.data.sum(axis=1)
    tes5 = tes4.mean()
    return tes5




def RFtree( train_matrix, tsmat, od, fn):
    #nos = [3, 5, 7]
    sample_cpt_dict = {}
    tset_llh = []
    print("evaluating the loglikelihood on test set 10 times")
    for n in range(10):
        print("evaluation no. : {}".format(n + 1))
        samples = []
        # generating the samples
        for k in range(od[fn][0]):
            temp_arr = np.array([np.inf] *  train_matrix.shape[1])
            for i in range(int( train_matrix.shape[0] / 3)):
                ri = np.random.randint(0,  train_matrix.shape[0])
                temp_arr = np.vstack((temp_arr,  train_matrix[ri]))
            temp_arr = temp_arr[1:, :]
            samples.append(temp_arr)
        # making the chow_liu_algo trees off the samples generated
        cpt_list = []
        for s in range(len(samples)):
            pos_prob = samples[s].sum(axis=0)
            pos_prob = pos_prob + 1
            neg_prob = samples[s].shape[0] - pos_prob + 2
            pos_prob = np.true_divide(pos_prob, samples[s].shape[0] + 2)
            neg_prob = np.true_divide(neg_prob, samples[s].shape[0] + 2)
            Weight = np.zeros((samples[s].shape[1], samples[s].shape[1]))
            for col1 in range(int(samples[s].shape[1])):
                for col2 in range(int(samples[s].shape[1])):
                    if col1 == col2:
                        pass
                    else:
                        a00 = (samples[s][:, col1] == 0) & (samples[s][:, col2] == 0)
                        a01 = (samples[s][:, col1] == 0) & (samples[s][:, col2] == 1)
                        a10 = (samples[s][:, col1] == 1) & (samples[s][:, col2] == 0)
                        a11 = (samples[s][:, col1] == 1) & (samples[s][:, col2] == 1)
                        b00 = a00.sum()
                        b01 = a01.sum()
                        b10 = a10.sum()
                        b11 = a11.sum()
                        P00 = np.true_divide(b00 + 1, int(samples[s].shape[0]) + 4)
                        P01 = np.true_divide(b01 + 1, int(samples[s].shape[0]) + 4)
                        P10 = np.true_divide(b10 + 1, int(samples[s].shape[0]) + 4)
                        P11 = np.true_divide(b11 + 1, int(samples[s].shape[0]) + 4)
                        W = P00 * np.log(np.true_divide(P00, (neg_prob[col1] * neg_prob[col2])))
                        W = W + P01 * np.log(np.true_divide(P01, (neg_prob[col1] * pos_prob[col2])))
                        W = W + P10 * np.log(np.true_divide(P10, (pos_prob[col1] * neg_prob[col2])))
                        W = W + P11 * np.log(np.true_divide(P11, (pos_prob[col1] * pos_prob[col2])))
                        Weight[col1][col2] = W
            for num in range(od[fn][1]):
                ri1 = np.random.randint(0,  train_matrix.shape[1])
                ri2 = np.random.randint(0,  train_matrix.shape[1])
                Weight[ri1][ri2] = 0
                Weight[ri2][ri1] = 0
            Weight = Weight * (-1)
            e = csr_matrix(Weight)
            f = minimum_spanning_tree(e).toarray()
            f_csr = csr_matrix(f)
            l1, l2 = f_csr.toarray().nonzero()
            edges = zip(l1, l2)
            graph = Graph()
            for e in edges:
                graph.addEdge(e[0], e[1])
            # The 0th feature is chosen as the root node
            graph.DFS(0, Weight.shape[0])
            o = graph.order
            pa = {o[0]: np.nan}
            for i in range(1, len(o)):
                if o[i] in graph.graph[o[i - 1]]:
                    pa[o[i]] = o[i - 1]
                else:
                    for j in range(i - 1):
                        if o[i] in graph.graph[o[i - j - 2]]:
                            pa[o[i]] = o[i - j - 2]
                            break
                        else:
                            pass

            cpt_matrix = []
            for child in list(pa.keys())[1:]:
                A00 = (samples[s][:, child] == 0) & (samples[s][:, pa[child]] == 0)
                A01 = (samples[s][:, child] == 1) & (samples[s][:, pa[child]] == 0)
                A10 = (samples[s][:, child] == 0) & (samples[s][:, pa[child]] == 1)
                A11 = (samples[s][:, child] == 1) & (samples[s][:, pa[child]] == 1)
                B00 = A00.sum()
                B01 = A01.sum()
                B10 = A10.sum()
                B11 = A11.sum()
                # temp1 = (samples[s][:, pa[child]] == 0).sum()
                # temp2 = (samples[s][:, pa[child]] == 1).sum()
                p00 = np.true_divide(B00 + 1, (samples[s][:, pa[child]] == 0).sum() + 2)
                p01 = np.true_divide(B01 + 1, (samples[s][:, pa[child]] == 0).sum() + 2)
                p10 = np.true_divide(B10 + 1, (samples[s][:, pa[child]] == 1).sum() + 2)
                p11 = np.true_divide(B11 + 1, (samples[s][:, pa[child]] == 1).sum() + 2)
                cpt_matrix.append([p00, p01, p10, p11])
            cpt_matrix = np.array(cpt_matrix)
            lcpt_matrix = np.log(cpt_matrix)
            # log of probabilities of the first feature which is the root
            pos_prob_X0 = np.log(np.true_divide((samples[s][:, 0].sum() + 1), samples[s].shape[0] + 2))
            neg_prob_X0 = np.log(
                np.true_divide((samples[s].shape[0] - samples[s][:, 0].sum() + 1), samples[s].shape[0] + 2))
            lcpt_matrix = np.vstack(([pos_prob_X0, neg_prob_X0, 0, 0], lcpt_matrix))
            cpt_list.append(lcpt_matrix)
        sample_cpt_dict[od[fn][0]] = cpt_list

        f_llh_sum = []

        for num in range(od[fn][0]):
            t = tsmat.copy()
            t = t.astype(float)
            for i in range(1, tsmat.shape[1]):
                par = pa[i]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 0), i] = sample_cpt_dict[od[fn][0]][num][i][0]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 0), i] = sample_cpt_dict[od[fn][0]][num][i][1]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 1), i] = sample_cpt_dict[od[fn][0]][num][i][2]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 1), i] = sample_cpt_dict[od[fn][0]][num][i][3]
            # filling up the 0th column in t with the respective probabilities
            t[(tsmat[:, 0]) == 0, 0] = sample_cpt_dict[od[fn][0]][num][0][1]
            t[(tsmat[:, 0]) == 1, 0] = sample_cpt_dict[od[fn][0]][num][0][0]
            llh = t.sum(axis=1)
            llh = llh.mean()
        f_llh_sum.append(np.true_divide(llh, od[fn][0]))
        f_llh_sum = np.array(f_llh_sum)
        f_llh = f_llh_sum.sum()
        tset_llh.append(f_llh)
    tset_llh = np.array(tset_llh)
    t1 = tset_llh.mean()
    t2 = tset_llh.std()
    return tset_llh, t1, t2



def MT(tmat, tsmat, od, fn):
    train_matrix = tmat.copy()
    #contains all cpts of each k in a list
    K_cpts = {}
    #contains Lambdas of each K ina a list
    K_lambdas = {}
    #storing the results of 10 test set log likelihoods
    tset_llh = []

    print("Testing till 10 iterations")
    for i in range(10):
        test_llh_sum = []
        print("{}th evaluation".format(i))
        #E step
        h_mat = np.random.rand( train_matrix.shape[0], od[fn])
        hmat_l = []
        llhs_list = []
        cpts_l = []
        # making the mutual infomation matrix
        # M step
        epochs = 50
        for i in range(epochs):
            if (i + 1) % 10 == 0:
                print("iteration : {}".format(i + 1))
            h_mat = h_mat / h_mat.sum(axis=1)[:, None]
            hmat_l.append(h_mat.copy())
            llh_list = []
            cpt_l = []
            for k in range(od[fn]):
                l = np.true_divide(h_mat[:, k].sum(), h_mat.shape[0])
                Weight = np.zeros(( train_matrix.shape[1],  train_matrix.shape[1]))
                for col1 in range(int( train_matrix.shape[1])):
                    for col2 in range(int( train_matrix.shape[1])):
                        if col1 == col2:
                            pass
                        else:
                            a00 = h_mat[:, k][( train_matrix[:, col1] == 0) & ( train_matrix[:, col2] == 0)]
                            a01 = h_mat[:, k][( train_matrix[:, col1] == 0) & ( train_matrix[:, col2] == 1)]
                            a10 = h_mat[:, k][( train_matrix[:, col1] == 1) & ( train_matrix[:, col2] == 0)]
                            a11 = h_mat[:, k][( train_matrix[:, col1] == 1) & ( train_matrix[:, col2] == 1)]
                            b00 = a00.sum()
                            b01 = a01.sum()
                            b10 = a10.sum()
                            b11 = a11.sum()

                            P00 = np.true_divide(b00 + l, h_mat[:, k].sum() + 4 * l)
                            P01 = np.true_divide(b01 + l, h_mat[:, k].sum() + 4 * l)
                            P10 = np.true_divide(b10 + l, h_mat[:, k].sum() + 4 * l)
                            P11 = np.true_divide(b11 + l, h_mat[:, k].sum() + 4 * l)

                            pos_prob0 = h_mat[:, k][ train_matrix[:, col1] == 1]
                            pos_prob0 = np.true_divide(pos_prob0.sum() + l, h_mat[:, k].sum() + 2 * l)
                            pos_prob1 = h_mat[:, k][ train_matrix[:, col2] == 1]
                            pos_prob1 = np.true_divide(pos_prob1.sum() + l, h_mat[:, k].sum() + 2 * l)
                            neg_prob0 = h_mat[:, k][ train_matrix[:, col1] == 0]
                            neg_prob0 = np.true_divide(neg_prob0.sum() + l, h_mat[:, k].sum() + 2 * l)
                            neg_prob1 = h_mat[:, k][ train_matrix[:, col2] == 0]
                            neg_prob1 = np.true_divide(neg_prob1.sum() + l, h_mat[:, k].sum() + 2 * l)

                            W = P00 * np.log(np.true_divide(P00, (neg_prob0 * neg_prob1)))
                            W = W + P01 * np.log(np.true_divide(P01, (neg_prob0 * pos_prob1)))
                            W = W + P10 * np.log(np.true_divide(P10, (pos_prob0 * neg_prob1)))
                            W = W + P11 * np.log(np.true_divide(P11, (pos_prob0 * pos_prob1)))
                            Weight[col1][col2] = W
                Weight = Weight * (-1)
                e = csr_matrix(Weight)
                f = minimum_spanning_tree(e).toarray().astype(float)
                f_csr = csr_matrix(f)
                l1, l2 = f_csr.toarray().nonzero()
                edges = zip(l1, l2)
                graph = Graph()
                for e in edges:
                    graph.addEdge(e[0], e[1])
                # The 0th feature is chosen as the root node
                graph.DFS(0, Weight.shape[0])
                o = graph.order
                pa = {o[0]: np.nan}
                for i in range(1, len(o)):
                    if o[i] in graph.graph[o[i - 1]]:
                        pa[o[i]] = o[i - 1]
                    else:
                        for j in range(i - 1):
                            if o[i] in graph.graph[o[i - j - 2]]:
                                pa[o[i]] = o[i - j - 2]
                                break
                            else:
                                pass

                cpt_matrix = []

                for child in list(pa.keys())[1:]:
                    try:
                        lc0 = np.true_divide((h_mat[:, k][ train_matrix[:, pa[child]] == 0]).sum(), ( train_matrix[:, pa[child]] == 0).sum())
                    except RuntimeWarning:
                        lc0 = 0.5
                    try:
                        lc1 = np.true_divide(h_mat[:, k][ train_matrix[:, pa[child]] == 1].sum(), ( train_matrix[:, pa[child]] == 1).sum())
                    except RuntimeWarning:
                        lc1 = 0.5
                    A00 = h_mat[:, k][( train_matrix[:, child] == 0) & ( train_matrix[:, pa[child]] == 0)]
                    A00 = np.true_divide(A00.sum() + lc0, h_mat[:, k][ train_matrix[:, pa[child]] == 0].sum() + 2 * lc0)
                    A01 = h_mat[:, k][( train_matrix[:, child] == 1) & ( train_matrix[:, pa[child]] == 0)]
                    A01 = np.true_divide(A01.sum() + lc0, h_mat[:, k][ train_matrix[:, pa[child]] == 0].sum() + 2 * lc0)
                    A10 = h_mat[:, k][( train_matrix[:, child] == 0) & ( train_matrix[:, pa[child]] == 1)]
                    A10 = np.true_divide(A10.sum() + lc1, h_mat[:, k][ train_matrix[:, pa[child]] == 1].sum() + 2 * lc1)
                    A11 = h_mat[:, k][( train_matrix[:, child] == 1) & ( train_matrix[:, pa[child]] == 1)]
                    A11 = np.true_divide(A11.sum() + lc1, h_mat[:, k][ train_matrix[:, pa[child]] == 1].sum() + 2 * lc1)
                    cpt_matrix.append([A00, A01, A10, A11])

                cpt_matrix = np.array(cpt_matrix)
                pos_prob_X0 = h_mat[:, k][( train_matrix[:, 0] == 1)]
                pos_prob_X0 = np.log(np.true_divide(pos_prob_X0.sum() + l, h_mat[:, k].sum() + 2 * l))

                neg_prob_X0 = h_mat[:, k][( train_matrix[:, 0] == 0)]
                neg_prob_X0 = np.log(np.true_divide(neg_prob_X0.sum() + l, h_mat[:, k].sum() + 2 * l))

                lcpt_matrix = np.log(cpt_matrix)
                lcpt_matrix = np.vstack((np.array([pos_prob_X0, neg_prob_X0, 0, 0]), lcpt_matrix))
                cpt_l.append(lcpt_matrix)

                t =  train_matrix.copy()
                t = t.astype(float)
                for i in range(1,  train_matrix.shape[1]):
                    par = pa[i]
                    t[( train_matrix[:, i] == 0) & ( train_matrix[:, par] == 0), i] = lcpt_matrix[i][0]
                    t[( train_matrix[:, i] == 1) & ( train_matrix[:, par] == 0), i] = lcpt_matrix[i][1]
                    t[( train_matrix[:, i] == 0) & ( train_matrix[:, par] == 1), i] = lcpt_matrix[i][2]
                    t[( train_matrix[:, i] == 1) & ( train_matrix[:, par] == 1), i] = lcpt_matrix[i][3]
                # filling up the 0th column in t with the respective probabilities
                t[( train_matrix[:, 0]) == 0, 0] = lcpt_matrix[0][1]
                t[( train_matrix[:, 0]) == 1, 0] = lcpt_matrix[0][0]
                # summing the logs to find the loglikelihood
                LLH_col = t.sum(axis=1)
                llh_list.append(LLH_col.mean())
                Phgx = h_mat[:, k] * np.exp(LLH_col)
                if i == epochs - 1:
                    pass
                else:
                    h_mat[:, k] = Phgx
            cpts_l.append(cpt_l)
            llhs_list.append(llh_list)
            K_cpts[od[fn]] = cpts_l[-1]

            temp_l = [np.true_divide(h_mat[:, hcol].sum(), h_mat.shape[0]) for hcol in range(od[fn])]

            K_lambdas[od[fn]] = temp_l
        for k in range(od[fn]):
            #summation of lambda k multiplied by the tree
            t = tsmat.copy()
            t = t.astype(float)
            for i in range(1, tsmat.shape[1]):
                par = pa[i]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 0), i] = K_cpts[od[fn]][k][i][0]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 0), i] = K_cpts[od[fn]][k][i][1]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 1), i] = K_cpts[od[fn]][k][i][2]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 1), i] = K_cpts[od[fn]][k][i][3]
            # filling up the 0th column in t with the respective probabilities
            t[(tsmat[:, 0]) == 0, 0] = K_cpts[od[fn]][k][0][1]
            t[(tsmat[:, 0]) == 1, 0] = K_cpts[od[fn]][k][0][0]
            llh = t.sum(axis = 1)
            llh = llh.mean()
            test_llh_sum.append(llh * K_lambdas[od[fn]][k])
        test_llh_sum = np.array(test_llh_sum)
        test_mllh = test_llh_sum.sum()
        tset_llh.append(test_mllh)
    tsetllh = np.array(tset_llh)
    t1 = np.mean(tset_llh)
    t2 = np.std(tset_llh)

    return tsetllh, t1, t2




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm_number', '--algo_no', type=int)
    parser.add_argument('-train_data', '--train_set', type=str)
    parser.add_argument('-valid_data', '--valid_set', type=str)
    parser.add_argument('-test_data', '--test_set', type=str)

    arg = parser.parse_args()
    algo_no = arg.algo_no
    train_name = arg.train_set
    valid_name = arg.valid_set
    test_name = arg.test_set
    
    #fnames = ["accidents", "baudio", "bnetflix", "dna", "jester", "kdd", "msnbc", "nltcs", "plants", "r52"]
    train_matrix = np.loadtxt(train_name, delimiter=',')
    valid_matrix = np.loadtxt(valid_name, delimiter=',')
    test_matrix = np.loadtxt(test_name, delimiter=',')

    if algo_no == 1:
            #Independent bayesian networks algo
            print("Executing Independent Bayesian Networks")
            avgLL_IBN_algo = IBN_algo(train_matrix, valid_matrix, test_matrix)
            #print("fname = {}".format(fname))
            print("Average log-likelihood for independent bayesian networks : ", avgLL_IBN_algo)
    elif algo_no == 2:
            #Chow-liu tree
            print("Executing Chow Liu Tree")
            avgLL_cl = chow_liu_algo(train_matrix, valid_matrix, test_matrix)
            #print("fname = {}".format(fname))
            print("Average log-likelihood for chow_liu_algo : ", avgLL_cl)
    elif algo_no == 3:
            #Mixture of trees
            fname = train_name.split("/")[-1]
            fname = fname.split(".")[0]
            print("Executing Mixture of Trees")
            print("fname = {}".format(fname))
            optimal_k_dict = {"accidents" : 4, "baudio" : 3, "bnetflix" : 2, "dna" : 3, "jester" : 3, "kdd" : 2, "msnbc" : 3, "nltcs" : 5, "plants" : 3, "r52" : 2}
            MTllh_list, MTaverage, MTstd = MT(train_matrix, test_matrix, optimal_k_dict, "nltcs")
            print("llh list : {}".format(MTllh_list))
            print("average: {}".format(MTaverage))
            print("std: {}".format(MTstd))
    elif algo_no == 4:
            #rf tree
            fname = train_name.split("/")[-1]
            fname = fname.split(".")[0]
            print("Executing Random Forest")
            print("fname = {}".format(fname))
            optimal_kr_dict = {"accidents": [4, 5], "baudio": [3, 6], "bnetflix": [2, 5], "dna": [3, 6], "jester": [3, 5],
                               "kdd": [2, 7], "msnbc": [3, 6], "nltcs": [5, 5], "plants": [2, 5], "r52": [2, 6]}
            RF_avgllh, RFavg, RFsd = RFtree(train_matrix,test_matrix, optimal_kr_dict, fname)
            print("Random Forest style mixture of trees {}".format(RF_avgllh))
            print("Average LogLikelihood for RF", RFavg)
            print("Standard deviation = {}".format(RFsd))
    
main()




import geatpy as ea
import random
import warnings
import numpy as np
import copy

help(ea.pbi)


def tcheby(ObjV, weights, idealPoint, CV=None, maxormins=None):
    num, M = ObjV.shape
    newobjv = copy.deepcopy(ObjV)
    if maxormins is None:
        maxormins = np.ones(M)
    for i in range(M):
        if maxormins[i] == 1:
            newobjv[:, i] = -newobjv[:, i]
    combinobjv = []
    if num == 1:
        num = weights.shape[0]
        for i in range(num):
            combinobjv.append(caucalatetcheby(ObjV[0], weights[i], idealPoint))
    else:
        assert num == weights.shape[0], " 当ObjV有多行时，weights的行数必须和ObjV的相等。"
        for i in range(num):
            combinobjv.append(caucalatetcheby(ObjV[i], weights[i], idealPoint))
    combinobjv = np.array(combinobjv).reshape((-1, 1))
    # print(combinobjv)
    # if CV is not None:
    #     num, M = CV.shape
    #     maxone = np.max(combinobjv)
    #     for i in range(num):
    #         tmpsum = 0
    #         for k in range(M):
    #             if CV[i, k] > 0:
    #                 tmpsum = tmpsum + CV[i, k]
    #         if tmpsum > 0:
    #             combinobjv[i] = maxone + tmpsum
    return combinobjv


def caucalatetcheby(objv, weight, idealpoint):
    # print(objv,weight,idealpoint)
    return np.max(abs(objv - idealpoint) * weight)

def pbi(ObjV, weights, idealPoint, CV=None, maxormins=None,theta=5):
    num, M = ObjV.shape
    newobjv = copy.deepcopy(ObjV)
    if maxormins is None:
        maxormins = np.ones(M)
    for i in range(M):
        if maxormins[i] == 1:
            newobjv[:, i] = -newobjv[:, i]
    combinobjv = []
    if num == 1:
        num = weights.shape[0]
        for i in range(num):
            combinobjv.append(caucalatePBI(ObjV[0].reshape(1,-1), weights[i].reshape(-1,1), idealPoint.reshape(1,-1),theta))
    else:
        assert num == weights.shape[0], " 当ObjV有多行时，weights的行数必须和ObjV的相等。"
        for i in range(num):
            combinobjv.append(caucalatePBI(ObjV[i].reshape(1,-1), weights[i].reshape(-1,1), idealPoint.reshape(1,-1),theta))
    combinobjv = np.array(combinobjv).reshape((-1, 1))
    return combinobjv

def caucalatePBI(objv, weight, idealpoint,theta):
    # print(np.dot((objv - idealpoint),weight),np.linalg.norm(np.dot((objv - idealpoint),weight), ord=2))
    d1=np.dot((objv - idealpoint),weight)/np.linalg.norm(weight, ord=2)
    d2=np.linalg.norm(objv-(idealpoint+((d1*weight)/np.linalg.norm(weight)).T), ord=2)
    return d1+theta*d2
if __name__ == '__main__':
    m = 5
    dim = 5
    tmp = np.random.rand(dim * m).reshape((-1, m))
    # lmad=np.random.rand(10)
    lmad = np.ones((dim, m))
    idealpoint = np.array([i + 100 for i in range(m)])
    cv = np.array([[i - 3, i - 2, i - 6, -1] for i in range(dim)])
    # print(tmp, lmad, cv, idealpoint,cv)
    # print(tcheby(tmp, lmad, idealpoint,cv))
    # print(ea.tcheby(tmp, lmad, idealpoint,cv))
    # print(tcheby(tmp, lmad, idealpoint,None),ea.tcheby(tmp, lmad, idealpoint,None))
    print(pbi(tmp, lmad, idealpoint, None), ea.pbi(tmp, lmad, idealpoint, None))
    # print(caucalatetcheby(tmp,lmad,idealpoint))
    # print(tmp)
    # print(ea.)
    # k=
    # print(tmp[tmp>1])

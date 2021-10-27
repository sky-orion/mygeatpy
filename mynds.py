import copy
import random

import geatpy
import numpy as np
import matplotlib.pyplot as plt


class NSGA2core:
    def __init__(self, pop_size, pop_obj, maxormins=None):
        self.pop_size = pop_size
        self.pop_obj = pop_obj
        self.num_obj = pop_obj.shape[1]
        self.f = []
        self.sp = [[] for _ in range(pop_size)]
        self.np = np.zeros([pop_size, 1], dtype=int)
        self.rank = 1 / np.zeros([pop_size, 1], dtype=int)
        self.cd = np.zeros([pop_size, 1])
        self.maxormins = maxormins
        if self.maxormins is not None:
            newobj = copy.deepcopy(self.pop_obj)
            for i in range(self.maxormins.shape[0]):
                if (self.maxormins[i] == -1):
                    newobj[:, i] = -newobj[:, i]
            self.pop_obj = newobj
        self.feasable = np.zeros(pop_size)
        self.zstar = np.zeros(pop_obj.shape[1])
        self.theta = 10 * np.pi / (2 * pop_size)
        self.pf = 0
        self.usangle = False
        self.needNum = pop_size

    def __index(self, i, ):
        return np.delete(range(self.pop_size), i)

    def createz(self, pop_obj):
        self.zstar = pop_obj.min(0)

    def is_dominate(self, obj_a, obj_b, num_obj, i, j):  # a dominates b
        if self.feasable[i] == 0 and self.feasable[j] == 0:
            if type(obj_a) is not np.ndarray:
                obj_a, obj_b = np.array(obj_a), np.array(obj_b)
            res = np.array([np.sign(k) for k in obj_a - obj_b])
            res_ngt0, res_eqf1 = np.argwhere(res <= 0), np.argwhere(res == -1)
            if res_ngt0.shape[0] == num_obj and res_eqf1.shape[0] > 0:
                return True
            return False
        else:
            if (self.usangle):
                angles = self.angle(obj_a, obj_b)
                if (angles <= self.theta):
                    return True if self.feasable[i] < self.feasable[j] else False
                else:
                    r = random.random()
                    if (r < self.pf):
                        res = np.array([np.sign(k) for k in obj_a - obj_b])
                        res_ngt0, res_eqf1 = np.argwhere(res <= 0), np.argwhere(res == -1)
                    if res_ngt0.shape[0] == num_obj and res_eqf1.shape[0] > 0:
                        return True
                    return False
            else:
                return True if self.feasable[i] < self.feasable[j] else False

    def angle(self, x1, x2):
        normed1 = np.linalg.norm(x1 - self.zstar, ord=2)
        normed2 = np.linalg.norm(x2 - self.zstar, ord=2)
        return (np.arccos(
            np.matmul((x1 - self.zstar).reshape(1, -1), (x2 - self.zstar).reshape(-1, 1)) / (normed1 * normed2)))[0, 0]

    def __is_dominate(self, i, j, obj):
        return self.is_dominate(obj[i], obj[j], obj.shape[1], i, j)

    def __f1_dominate(self, ):
        f1 = []
        if self.pop_size == self.needNum:
            for i in range(self.pop_size):
                for j in self.__index(i):
                    if self.__is_dominate(i, j, self.pop_obj):
                        if j not in self.sp[i]:
                            self.sp[i].append(j)
                    elif self.__is_dominate(j, i, self.pop_obj):
                        self.np[i] += 1
                if self.np[i] == 0:
                    self.rank[i] = 1
                    f1.append(i)
        else:
            while (len(f1) != self.needNum):
                point = np.random.choice(self.pop_size, self.needNum * 2, replace=False)
                count = 0
                for i in point:
                    for j in range(count, point.shape[0]):
                        if self.__is_dominate(i, j, self.pop_obj):
                            if j not in self.sp[i]:
                                self.sp[i].append(j)
                        elif self.__is_dominate(j, i, self.pop_obj):
                            self.np[i] += 1
                    if self.np[i] == 0:
                        self.rank[i] = 1
                        f1.append(i)
                        if len(f1) == self.needNum:
                            break
                    count = count + 1
        # print(f1)
        return f1

    def ini(self, pop_size, num_obj):
        self.pop_size = pop_size
        self.num_obj = num_obj
        self.sp = [[] for _ in range(pop_size)]
        self.np = np.zeros([pop_size, 1], dtype=int)
        self.rank = 1 / np.zeros([pop_size, 1], dtype=int)
        self.cd = np.zeros([pop_size, 1])
        self.feasable = np.zeros(pop_size)

    def fast_non_dominate_sort(self, pop_obj=None, neednum=None, needLevel=None, cv=None, maxormin=None, usangle=False):

        if pop_obj is not None:
            pop_size = pop_obj.shape[0]
            self.ini(pop_size, pop_obj.shape[1])
            self.pop_obj = pop_obj
        if neednum is not None:
            self.needNum = neednum
        if maxormin is not None:
            self.maxormins = maxormin
            newobj = copy.deepcopy(self.pop_obj)
            for i in range(self.maxormins.shape[0]):
                if (self.maxormins[i] == -1):
                    newobj[:, i] = -newobj[:, i]
            self.pop_obj = newobj
        if cv is not None:
            for t in range(cv.shape[0]):
                ok = 0
                for j in range(cv.shape[1]):
                    if (cv[t, j] > 0):
                        ok = cv[t, j] + ok
                        continue
                if ok != 0:
                    # print(t,cv.shape)
                    self.feasable[t] = ok
            if (usangle):
                self.usangle = usangle
                self.pf = sum(self.feasable <= 0) / self.pop_size
                self.createz(pop_obj)

        rank = 1
        f1 = self.__f1_dominate()
        # print(f1)
        while f1:
            self.f.append(f1)
            q = []
            for i in f1:
                for j in self.sp[i]:
                    self.np[j] -= 1
                    if self.np[j] == 0:
                        self.rank[j] = rank + 1
                        q.append(j)
            rank += 1
            f1 = q
        return self.rank.reshape(-1), 0

    def sort_obj_by(self, front=None, j=0, ):
        if front is not None:
            index = np.argsort(self.pop_obj[front, j])
        else:
            index = np.argsort(self.pop_obj[:, j])
        return index

    def crowd_distance(self, pop_obj=None):
        if (pop_obj is not None):
            self.pop_obj = pop_obj
        for f in self.f:
            len_f1 = len(f) - 1

            for j in range(self.num_obj):
                index = self.sort_obj_by(f, j)
                # print(index,self.pop_obj[f, j])
                sorted_obj = self.pop_obj[f][index]
                # print( sorted_obj[-1, j] - sorted_obj[0, j],self.pop_obj[f[index[0]],j]- self.pop_obj[f[index[-1]],j])
                obj_range_fj = self.pop_obj[f[index[-1]], j] - self.pop_obj[f[index[0]], j]
                if(np.isnan(obj_range_fj) or obj_range_fj==0):
                    obj_range_fj=1
                # obj_range_fj = sorted_obj[-1, j] - sorted_obj[0, j]
                self.cd[f[index[0]]] = np.inf
                self.cd[f[index[-1]]] = np.inf
                for i in range(1, index.shape[0] - 1):
                    self.cd[i] = self.cd[i] + (abs(self.pop_obj[f[index[i + 1]], j] - self.pop_obj[f[index[i - 1]], j])) / obj_range_fj
                    # print((self.pop_obj[f[index[i + 1]], j] - self.pop_obj[f[index[i - 1]], j]))
                    # k = np.argwhere(np.array(f)[index] == i)[:, 0][0]
                    # if 0 < index[k] < len_f1:
                    #     self.cd[i] += (sorted_obj[index[k] + 1, j] - sorted_obj[index[k] - 1, j]) / obj_range_fj

        # for f in self.f:
        #     len_f1 = len(f) - 1
        #     for j in range(self.num_obj):
        #         index = self.sort_obj_by(f, j)
        #         sorted_obj = self.pop_obj[f][index]
        #         obj_range_fj = sorted_obj[-1, j] - sorted_obj[0, j]
        #         self.cd[f[index[0]]] = np.inf
        #         self.cd[f[index[-1]]] = np.inf
        #         for i in f:
        #             k = np.argwhere(np.array(f)[index] == i)[:, 0][0]
        #             if 0 < index[k] < len_f1:
        #                 self.cd[i] += (sorted_obj[index[k] + 1, j] - sorted_obj[index[k] - 1, j]) / obj_range_fj
        return self.cd.reshape(-1)


if __name__ == '__main__':
    pop_obj = np.array([
        [2, 7.5],
        [3, 6],
        [3, 7.5],
        [4, 5],
        [4, 6.5],
        [5, 4.5],
        [5, 6],
        [5, 7],
        [6, 6.5],
        [2, 7.5],
        [3, 6],
        [3, 7.5],
        [4, 5],
        [4, 6.5],
        [5, 4.5],
        [5, 6],
        [5, 7],
        [6, 6.5],
        [2, 7.5],
        [3, 6],
        [3, 7.5],
        [4, 5],
        [4, 6.5],
        [5, 4.5],
        [5, 6],
        [5, 7],
        [6, 6.5],
        [2, 7.5],
        [3, 6],
        [3, 7.5],
        [4, 5],
        [4, 6.5],
        [5, 4.5],
        [5, 6],
        [5, 7],
        [6, 6.5],
        [2, 7.5],
        [3, 6],
        [3, 7.5],
        [4, 5],
        [4, 6.5],
        [5, 4.5],
        [5, 6],
        [5, 7],
        [6, 6.5],
        [1, 7.5],
        [2, 6],
        [2, 7.5],
        [3, 5],
        [3, 6.5],
        [4, 4.5],
        [6, 6],
        [7, 7],
        [8, 6.5],
    ])
    maxormin = np.array([-1, 1])
    NSGA2core = NSGA2core(pop_obj.shape[0], pop_obj, None)
    # print(pareto.fast_non_dominate_sort())
    # pareto.crowd_distance()
    print("# rank\n", NSGA2core.fast_non_dominate_sort())
    print("# front\n", NSGA2core.f)
    print("# crowd distance\n", NSGA2core.crowd_distance())
    print(geatpy.ndsortESS(pop_obj, None, None, None, maxormin)[0])
    print("test", geatpy.crowdis(pop_obj, geatpy.ndsortTNS(pop_obj, None, None, None, None)[0]))
    for i, f in enumerate(NSGA2core.f):
        plt.plot(pop_obj[f, 0], pop_obj[f, 1], "o--", label="${Rank}-{%s}$" % (i + 1))
    plt.legend()
    plt.savefig("Pareto.png")
    plt.show()

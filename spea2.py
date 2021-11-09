# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from scipy.spatial.distance import cdist
from sys import path as paths
from os import path

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class moea_SPEA2_templet(ea.MoeaAlgorithm):

    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'SPEA2'
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
        self.archive = None
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择

    def Fitnessassignment(self, population, k):
        objective_values = population.ObjV
        equals = np.equal(objective_values[:, np.newaxis, :], objective_values[np.newaxis, :, :])
        dif = objective_values[:, np.newaxis, :] >= objective_values[np.newaxis, :, :]
        dif = np.count_nonzero(dif, axis=2)
        dif = dif // objective_values.shape[1]
        equals = np.count_nonzero(equals, axis=2)
        equals = equals // objective_values.shape[1]
        equals = 1 - equals
        dominated = np.logical_and(dif, equals)
        S = dominated.sum(axis=0)
        R = S * dominated
        R = R.sum(axis=1)
        distances = (objective_values[:, np.newaxis, :] - objective_values[np.newaxis, :, :]) ** 2
        distances = np.sort(np.sqrt(distances.sum(axis=2)), axis=1)
        k_distance = distances[:, k]
        D = 1 / (k_distance + 2)
        # Fitness
        return D.reshape((-1, 1)) + R.reshape((-1, 1))

    def EnvironmentalSelection(self, totalpopulation, N_archive):
        objective_values = totalpopulation.ObjV
        F = totalpopulation.FitnV
        indices_next_archive = np.where(F < 1)[0]
        length = len(indices_next_archive)
        if length < N_archive:
            best_inidices = np.argsort(F.reshape(-1), kind="mergesort")
            indices_next_archive = best_inidices[:N_archive]
        elif length > N_archive:
            rest = length - N_archive
            dist = (objective_values[[indices_next_archive], np.newaxis, :] - objective_values[np.newaxis,
                                                                              [indices_next_archive], :]) ** 2
            dist = np.sqrt(np.sum(np.squeeze(dist), axis=2))
            ind = np.triu_indices(dist.shape[0])
            dist[ind] = np.inf
            y, _ = np.unravel_index(np.argsort(dist.ravel(), kind="mergesort"), dist.shape)
            _, i = np.unique(y, return_index=True)
            indices_to_delete = y[sorted(i)][:rest]
            indices_next_archive = np.delete(indices_next_archive, indices_to_delete)
        return totalpopulation[indices_next_archive]

    def terminated(self, pop):  # 判断是终止进化，pop为当代种群对象
        self.stat(pop)  # 进行统计分析，更新进化记录器
        if self.currentGen + 1 >= self.MAXGEN:
            return True
        else:
            self.currentGen += 1  # 进化代数+1
            return False

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = self.population.sizes
        N_archive = NIND
        K = int(np.sqrt(NIND + N_archive))  # 邻居数
        maxindice = np.where(self.problem.maxormins == -1)  # 为最大化函数的位置
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        population.initChrom(NIND)  # 初始化种群染色体矩阵，此时种群规模将调整为uniformPoint点集的大小，initChrom函数会把种群规模给重置
        self.call_aimFunc(population)  # 计算种群的目标函数值
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        population.ObjV[:, maxindice] = -population.ObjV[:, maxindice]
        totalpopulation = population
        # ===========================开始进化============================
        while True:
            totalpopulation.FitnV = self.Fitnessassignment(totalpopulation, K)
            self.archive = self.EnvironmentalSelection(totalpopulation, N_archive)
            offspring = self.archive[ea.selecting(self.selFunc, self.archive.FitnV, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            population = offspring
            totalpopulation = population + self.archive
            self.call_aimFunc(totalpopulation)
            totalpopulation.ObjV[:, maxindice] = -totalpopulation.ObjV[:, maxindice]
            # self.call_aimFunc(population)  # 计算种群的目标函数值
            # self.call_aimFunc(self.archive)  # 计算种群的目标函数值
            # population.ObjV[:, maxindice] = -population.ObjV[:, maxindice]
            # self.archive.ObjV[:, maxindice] = -self.archive.ObjV[:, maxindice]
            # totalpopulation = population + self.archive
            if self.terminated(self.archive):
                break

        return self.finishing(self.archive)  # 调用finishing完成后续工作并返回结果


if __name__ == '__main__':
    help(ea.ndsortESS)

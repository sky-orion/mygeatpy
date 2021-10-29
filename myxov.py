# -*- coding: utf-8 -*-
import copy
import numpy as np
from geatpy.core import xovpmx, mutinv, xovdp, mutbin
from geatpy.operators.recombination.Recombination import Recombination
from geatpy.operators.mutation.Mutation import *
from geatpy.core import mutbga
import geatpy
import random
import warnings

warnings.filterwarnings("ignore")


def crtfld(Encoding, problem_varTypes, problem_ranges, problem_borders, precisions=None, codes=0):
    assert Encoding == 'BG' or Encoding == 'RI' or Encoding == 'P', "编码必须为BG、RI、P"
    assert problem_varTypes.shape[0] == problem_ranges.shape[1] and problem_ranges.shape[1] == problem_borders.shape[
        1], "输入维度不一致"
    assert problem_ranges.shape[0] == 2 and problem_borders.shape[0] == 2, "range的维度必须为2*N"

    dim = problem_varTypes.shape[0]
    if precisions is None:
        precisions = [4.0] * dim
    if Encoding == 'BG':
        lens = [31] * dim  # 长度
        lb = problem_ranges[0]
        ub = problem_ranges[1]
        code = [codes] * dim  # 编码方式
        scales = [0] * dim
        lbin = problem_borders[0]
        ubin = problem_borders[1]
        varTypes = problem_varTypes
        field = np.stack([lens, lb, ub, code, scales, lbin, ubin, varTypes], axis=0)
        return field
    if Encoding == 'P':
        lb = problem_ranges[0]
        ub = problem_ranges[1]
        varTypes = [1] * dim
        lbin = problem_borders[0]
        ubin = problem_borders[1]
        for i in range(dim):
            if lbin[i] == 0:
                lb[i] = lb[i] + 1
            if ubin[i] == 0:
                ub[i] = ub[i] - 1
            assert lb[i] <= ub[i], "上界必须小于下界"
        field = np.stack([lb, ub, varTypes], axis=0)
        return field
    if Encoding == "RI":
        problem_ranges = problem_ranges.astype(np.float)
        lb = problem_ranges[0]
        ub = problem_ranges[1]
        lbin = problem_borders[0]
        ubin = problem_borders[1]
        for i in range(dim):
            if problem_varTypes[i] == 0:

                if lbin[i] == 0:
                    lb[i] = lb[i] + np.power(10, -1 * precisions[i])
                if ubin[i] == 0:
                    ub[i] = ub[i] - np.power(10, -1 * precisions[i])
                assert lb[i] <= ub[i], "上界必须小于下界"
            elif problem_varTypes[i] == 1:
                if lbin[i] == 0:
                    lb[i] = lb[i] + 1
                if ubin[i] == 0:
                    ub[i] = ub[i] - 1
                assert lb[i] <= ub[i], "上界必须小于下界"
        varTypes = problem_varTypes

        field = np.stack([lb, ub, varTypes], axis=0)
        return field


class Mutpolyn(Mutation):
    """
    Mutpolyn - class : 一个用于调用内核中的变异函数mutpolyn(多项式变异)的变异算子类，
                       该类的各成员属性与内核中的对应函数的同名参数含义一致，
                       可利用help(mutpolyn)查看各参数的详细含义及用法。
    """

    def __init__(self, Pm=None, DisI=20, FixType=1, Parallel=False):
        self.Pm = Pm  # 表示染色体上变异算子所发生作用的最小片段发生变异的概率
        self.DisI = DisI  # 多项式变异中的分布指数
        self.FixType = FixType  # 表示采用哪种方式来修复超出边界的染色体元素，可取值1，2，3，4，详细含义见help()帮助文档
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省时默认为False

    def do(self, Encoding, OldChrom, FieldDR, *args):  # 执行变异
        return self.mymutpolyn(Encoding, OldChrom, FieldDR, self.Pm, self.DisI, self.FixType, self.Parallel)

    def mymutpolyn(self, Encoding, OldChrom, FieldDR, Pm=None, DisI=20, FixType=1, Parallel=None):
        assert Encoding == "RI", "该函数只针对实整数值编码的种群进行变异，因此只允许Encoding为'RI'。"
        num, len = OldChrom.shape
        if Pm is None:
            Pm = 1 / len
        if DisI is None:
            DisI = 20
        for chrom in range(num):
            if (type(Pm) is float):
                for i in range(len):
                    prob = random.random()
                    if (prob <= Pm):
                        u = np.random.rand()
                        if u < 0.5:
                            beta = np.power(2 * u, 1 / (DisI + 1)) - 1
                        else:
                            beta = 1 - np.power(2 - 2 * u, 1 / (DisI + 1))
                        print(beta)
                        if FieldDR[2, i] == 1:

                            OldChrom[chrom, i] = round(OldChrom[chrom, i] + beta)
                        else:
                            OldChrom[chrom, i] = OldChrom[chrom, i] + beta

                        if (OldChrom[chrom, i] < FieldDR[0, i]) or (OldChrom[chrom, i] > FieldDR[1, i]):
                            OldChrom[chrom, i] = self.fix(OldChrom[chrom, i], FieldDR, FixType, i)
            else:
                for i in range(len):
                    prob = random.random()
                    if (prob <= Pm[i]):
                        u = random.random()
                        if u < 0.5:
                            beta = np.power(2 * u, 1 / (DisI + 1)) - 1
                        else:
                            beta = 1 - np.power(2 - 2 * u, 1 / (DisI + 1))
                        if FieldDR[2, i] == 1:
                            OldChrom[chrom, i] = round(OldChrom[chrom, i] + beta)
                        else:
                            OldChrom[chrom, i] = OldChrom[chrom, i] + beta
                        if (OldChrom[chrom, i] < FieldDR[0, i]) or (OldChrom[chrom, i] > FieldDR[1, i]):
                            OldChrom[chrom, i] = self.fix(OldChrom[chrom, i], FieldDR, FixType, i)
            # print(FieldDR,OldChrom)
            return OldChrom

    def fix(self, chrom, fleid, fixtype, point):
        lower = fleid[0, point]
        upper = fleid[1, point]
        vartype = fleid[2, point]
        # print(vartype)
        if vartype == 0:
            if chrom < lower:
                if fixtype == 1:
                    chrom = lower
                if fixtype == 2:
                    chrom = upper + chrom - lower
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 3:
                    chrom = 2 * lower - chrom
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 4:
                    r = random.random()
                    chrom = lower + (upper - lower) * r
            elif chrom > upper:
                if fixtype == 1:
                    chrom = upper
                if fixtype == 2:
                    chrom = lower + chrom - upper
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 3:
                    chrom = 2 * upper - chrom
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 4:
                    r = random.random()
                    chrom = lower + (upper - lower) * r
        else:
            if chrom < lower:
                if fixtype == 1:
                    chrom = lower
                if fixtype == 2:
                    chrom = upper + chrom - lower
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 3:
                    chrom = 2 * lower - chrom
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 4:
                    r = np.random.randint(lower, high=upper)
                    chrom = r
            elif chrom > upper:
                if fixtype == 1:
                    chrom = upper
                if fixtype == 2:
                    chrom = lower + chrom - upper
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 3:
                    chrom = 2 * upper - chrom
                    if chrom < lower or chrom > upper:
                        chrom = self.fix(chrom, fleid, 4, point)
                if fixtype == 4:
                    r = np.random.randint(lower, high=upper)
                    chrom = lower + (upper - lower) * r
        print(chrom)
        return chrom

    def getHelp(self):  # 查看内核中的变异算子的API文档
        help(geatpy.mutpolyn)


class Mutinv(Mutation):
    """
    Mutinv - class : 一个用于调用内核中的变异函数mutinv(染色体片段逆转变异)的变异算子类，
                     该类的各成员属性与内核中的对应函数的同名参数含义一致，
                     可利用help(mutinv)查看各参数的详细含义及用法。

    """

    def __init__(self, Pm=None, InvertLen=None, Parallel=False):
        self.Pm = Pm  # 表示染色体上变异算子所发生作用的最小片段发生变异的概率
        self.InvertLen = None  # 控制染色体发生反转的片段长度，当设置为None时取默认值，详见help(mutinv)帮助文档
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省时默认为False

    def do(self, Encoding, OldChrom, FieldDR, *args):  # 执行变异
        return self.mymutinv(Encoding, OldChrom, FieldDR, self.Pm, self.InvertLen, self.Parallel)

    def mymutinv(self, Encoding, OldChrom, FieldDR, Pm=1, InvertLen=None, Parallel=None):
        if len(OldChrom.shape) == 1:
            OldChrom = OldChrom.reshape((1, -1))
        num, lens = OldChrom.shape
        if Pm is None:
            Pm = 1
        if InvertLen is None:
            InvertLen = lens // 2
        if InvertLen > lens:
            InvertLen = lens
        re = []
        for chrom in range(num):
            prob = random.random()
            if (prob <= Pm):
                point = random.randint(0, lens - InvertLen)
                tmpchrom = OldChrom[chrom]
                newchrom = copy.deepcopy(tmpchrom)
                # print(point,InvertLen)
                if InvertLen == lens:
                    newchrom = tmpchrom[::-1]
                else:
                    # print(tmpchrom,point,InvertLen)
                    # point=0
                    if point == 0:
                        newchrom[point:point + InvertLen] = tmpchrom[point + InvertLen - 1::-1]
                    else:
                        newchrom[point:point + InvertLen] = tmpchrom[point + InvertLen - 1:point - 1:-1]
                re.append(newchrom)
            else:
                re.append(OldChrom[chrom])
        return np.array(re)

    def getHelp(self):  # 查看内核中的变异算子的API文档
        help(mutinv)


class Mutbin(Mutation):
    """
    Mutbin - class : 一个用于调用内核中的变异函数mutbin(二进制变异)的变异算子类，
                     该类的各成员属性与内核中的对应函数的同名参数含义一致，
                     可利用help(mutbin)查看各参数的详细含义及用法。

    """

    def __init__(self, Pm=None, Parallel=False):
        self.Pm = Pm  # 表示染色体上变异算子所发生作用的最小片段发生变异的概率
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省时默认为False

    def do(self, Encoding, OldChrom, FieldDR, *args):  # 执行变异
        return self.mymutbin(Encoding, OldChrom, Pm=self.Pm, Parallel=self.Parallel)

    def mymutbin(self, Encoding, OldChrom, Pm=None, Parallel=False):
        assert Encoding == "BG", "编码必须为BG"
        num, len = OldChrom.shape
        if Pm is None:
            Pm = 1 / len
        for chrom in range(num):
            if (type(Pm) is list):
                for i in range(len):
                    prob = random.random()
                    if (prob <= Pm[i]):
                        OldChrom[chrom, i] = 0 if OldChrom[chrom, i] == 1 else 1
            else:
                for i in range(len):
                    prob = random.random()
                    if (prob <= Pm):
                        OldChrom[chrom, i] = 0 if OldChrom[chrom, i] == 1 else 1
        return OldChrom

    def getHelp(self):  # 查看内核中的变异算子的API文档
        help(mutbin)


class Xovdp(Recombination):
    """
    Xovdp - class : 一个用于调用内核中的函数xovdp(两点交叉)的类，
                    该类的各成员属性与内核中的对应函数的同名参数含义一致，
                    可利用help(xovdp)查看各参数的详细含义及用法。

    """

    def __init__(self, XOVR=0.7, Half_N=False, GeneID=None, Parallel=False):
        self.XOVR = XOVR  # 发生交叉的概率
        self.Half_N = Half_N  # 该参数是旧版的输入参数Half的升级版，用于控制交配方式
        self.GeneID = GeneID  # 基因ID，是一个行向量，若设置了该参数，则该函数会对具有相同基因ID的染色体片段进行整体交叉
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省或为None时默认为False

    def do(self, OldChrom):  # 执行内核函数

        return self.myxovdp(OldChrom, self.XOVR, self.Half_N, self.GeneID, self.Parallel)

    def myxovdp(self, OldChrom, XOVR=0.7, Half_N=False, GeneID=None, Parallel=None):  # self总是指调用时的类的实例。
        if Half_N is False:  # Half_N = False时返回的NewChrom的行数与OldChrom一致,当Half_N为False时配对的两条染色体相互交叉返回两条染色体。
            if (type(OldChrom) is list):
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        prob = random.random()
                        if (prob <= XOVR):
                            chrom[i], chrom[i + num // 2] = self.two_points_cross(chrom[i], chrom[i + num // 2])
                return OldChrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    prob = random.random()
                    # print(prob)
                    if (prob <= XOVR):
                        chrom[i], chrom[i + num // 2] = self.two_points_cross(chrom[i], chrom[i + num // 2])
                return OldChrom
        elif Half_N is True:
            if (type(OldChrom) is list):
                newchrom = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        prob = random.random()
                        if (prob <= XOVR):
                            chrom[i], _ = self.two_points_cross(chrom[i], chrom[i + num // 2])
                    newchrom.append(chrom[:(num // 2)])
                return newchrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    prob = random.random()
                    if (prob <= XOVR):
                        chrom[i], _ = self.two_points_cross(chrom[i], chrom[i + num // 2])
                return OldChrom[:(num // 2)]
        else:
            if (type(OldChrom) is list):
                returnchroms = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    newchroms = []
                    for i in range(min(Half_N, num)):
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                        while point1 != point2:
                            point1 = random.randint(0, num - 1)
                            point2 = random.randint(0, num - 1)
                        nextchrom, _ = self.two_points_cross(chrom[point1], chrom[point2])
                        newchroms.append(nextchrom)
                    newchroms = np.array(newchroms)
                    returnchroms.append(newchroms)
                return returnchroms
            else:
                chrom = OldChrom
                num, len = chrom.shape
                newchroms = []
                for i in range(min(Half_N, num)):
                    point1 = random.randint(0, num - 1)
                    point2 = random.randint(0, num - 1)
                    while point1 != point2:
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                    nextchrom, _ = self.two_points_cross(chrom[point1], chrom[point2])
                    newchroms.append(nextchrom)
                newchroms = np.array(newchroms)
                return newchroms

    def two_points_cross(self, a1, a2):
        # 不改变原始数据进行操作
        newa1 = copy.deepcopy(a1)
        newa2 = copy.deepcopy(a2)
        # 交叉位置,point1<point2

        point1 = random.randint(0, newa1.shape[0] - 1)
        point2 = random.randint(0., newa1.shape[0] - 1)
        while point1 > point2 or point1 == point2:
            point1 = random.randint(0, newa1.shape[0] - 1)
            point2 = random.randint(0., newa1.shape[0] - 1)
        # 交叉
        newa1[point1:point2], newa2[point1:point2] = a2[point1:point2], a1[point1:point2]
        return newa1, newa2

    def getHelp(self):  # 查看内核中的重组算子的API文档
        help(xovdp)


class Xovud(Recombination):
    """
    Xovdp - class : 一个用于调用内核中的函数xovdp(两点交叉)的类，
                    该类的各成员属性与内核中的对应函数的同名参数含义一致，
                    可利用help(xovdp)查看各参数的详细含义及用法。

    """

    def __init__(self, XOVR=0.7, Half_N=False, GeneID=None, Parallel=False):
        self.XOVR = XOVR  # 发生交叉的概率
        self.Half_N = Half_N  # 该参数是旧版的输入参数Half的升级版，用于控制交配方式
        self.GeneID = GeneID  # 基因ID，是一个行向量，若设置了该参数，则该函数会对具有相同基因ID的染色体片段进行整体交叉
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省或为None时默认为False

    def do(self, OldChrom):  # 执行内核函数

        return self.myxovud(OldChrom, self.XOVR, self.Half_N, self.GeneID, self.Parallel)

    def myxovud(self, OldChrom, XOVR=0.7, Half_N=False, GeneID=None, Parallel=None):  # self总是指调用时的类的实例。
        if Half_N is False:  # Half_N = False时返回的NewChrom的行数与OldChrom一致,当Half_N为False时配对的两条染色体相互交叉返回两条染色体。
            if (type(OldChrom) is list):
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        chrom[i], chrom[i + num // 2] = self.two_points_cross(chrom[i], chrom[i + num // 2], XOVR)
                return OldChrom
            else:
                chrom = OldChrom
                print(chrom.shape)
                num, len = chrom.shape
                for i in range(num // 2):
                    chrom[i], chrom[i + num // 2] = self.two_points_cross(chrom[i], chrom[i + num // 2], XOVR)
                return OldChrom
        elif Half_N is True:
            if (type(OldChrom) is list):
                newchrom = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        chrom[i], _ = self.two_points_cross(chrom[i], chrom[i + num // 2], XOVR)
                    newchrom.append(chrom[:(num // 2)])
                return newchrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    chrom[i], _ = self.two_points_cross(chrom[i], chrom[i + num // 2], XOVR)
                return OldChrom[:(num // 2)]
        else:
            if (type(OldChrom) is list):
                returnchroms = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    newchroms = []
                    for i in range(min(Half_N, num)):
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                        while point1 != point2:
                            point1 = random.randint(0, num - 1)
                            point2 = random.randint(0, num - 1)
                        nextchrom, _ = self.two_points_cross(chrom[point1], chrom[point2], XOVR)
                        newchroms.append(nextchrom)
                    newchroms = np.array(newchroms)
                    returnchroms.append(newchroms)
                return returnchroms
            else:
                chrom = OldChrom
                num, len = chrom.shape
                newchroms = []
                for i in range(min(Half_N, num)):
                    point1 = random.randint(0, num - 1)
                    point2 = random.randint(0, num - 1)
                    while point1 != point2:
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                    nextchrom, _ = self.two_points_cross(chrom[point1], chrom[point2], XOVR)
                    newchroms.append(nextchrom)
                newchroms = np.array(newchroms)
                return newchroms

    def two_points_cross(self, a1, a2, XOVR=0.5):
        # print(a1.shape[0])
        for i in range(a1.shape[0]):
            prob = random.random()
            # print(prob,i,XOVR)
            if prob <= XOVR:
                tmp = (a2[i])
                a2[i] = a1[i]
                a1[i] = tmp
        return a1, a2

    def getHelp(self):  # 查看内核中的重组算子的API文档
        help(xovdp)


class Xovpmx(Recombination):
    """
    Xovpmx - class : 一个用于调用内核中的函数xovpmx(部分匹配交叉)的类，
                     该类的各成员属性与内核中的对应函数的同名参数含义一致，
                     可利用help(xovpmx)查看各参数的详细含义及用法。

    """

    def __init__(self, XOVR=0.7, Half_N=False, Method=1, Parallel=False):
        self.XOVR = XOVR  # 发生交叉的概率
        self.Half_N = Half_N  # 该参数是旧版的输入参数Half的升级版，用于控制交配方式
        self.Method = Method  # 表示部分匹配交叉采用什么方法进行交叉。
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省时默认为False

    def do(self, OldChrom):  # 执行内核函数

        return self.myxovpmx(OldChrom, self.XOVR, self.Half_N, self.Method, self.Parallel)

    def myxovpmx(self, OldChrom, XOVR=0.7, Half_N=False, GeneID=None, Parallel=None):
        if Half_N is False:  # Half_N = False时返回的NewChrom的行数与OldChrom一致,当Half_N为False时配对的两条染色体相互交叉返回两条染色体。
            if (type(OldChrom) is list):
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        prob = random.random()
                        if (prob <= XOVR):
                            chrom[i], chrom[i + num // 2] = self.two_points_cross(chrom[i], chrom[i + num // 2])
                return OldChrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    prob = random.random()
                    # print(prob)
                    if (prob <= XOVR):
                        chrom[i], chrom[i + num // 2] = self.two_points_cross(chrom[i], chrom[i + num // 2])
                return OldChrom
        elif Half_N is True:
            if (type(OldChrom) is list):
                newchrom = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        prob = random.random()
                        if (prob <= XOVR):
                            chrom[i], _ = self.two_points_cross(chrom[i], chrom[i + num // 2])
                    newchrom.append(chrom[:(num // 2)])
                return newchrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    prob = random.random()
                    if (prob <= XOVR):
                        chrom[i], _ = self.two_points_cross(chrom[i], chrom[i + num // 2])
                return OldChrom[:(num // 2)]
        else:
            if (type(OldChrom) is list):
                returnchroms = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    newchroms = []
                    for i in range(min(Half_N, num)):
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                        while point1 != point2:
                            point1 = random.randint(0, num - 1)
                            point2 = random.randint(0, num - 1)
                        nextchrom, _ = self.two_points_cross(chrom[point1], chrom[point2])
                        newchroms.append(nextchrom)
                    newchroms = np.array(newchroms)
                    returnchroms.append(newchroms)
                return returnchroms
            else:
                chrom = OldChrom
                num, len = chrom.shape
                newchroms = []
                for i in range(min(Half_N, num)):
                    point1 = random.randint(0, num - 1)
                    point2 = random.randint(0, num - 1)
                    while point1 != point2:
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                    nextchrom, _ = self.two_points_cross(chrom[point1], chrom[point2])
                    newchroms.append(nextchrom)
                newchroms = np.array(newchroms)
                return newchroms

    def getHelp(self):  # 查看内核中的重组算子的API文档
        help(xovpmx)

    def two_points_cross(self, a1, a2):
        # 不改变原始数据进行操作
        a1 = a1.tolist()
        a2 = a2.tolist()
        a1_1 = copy.deepcopy(a1)
        a2_1 = copy.deepcopy(a2)
        # 交叉位置,point1<point2
        point1 = random.randint(0, len(a1_1))
        point2 = random.randint(0., len(a1_1))
        while point1 > point2 or point1 == point2:
            point1 = random.randint(0, len(a1_1))
            point2 = random.randint(0., len(a1_1))
        # 记录交叉项
        fragment1 = a1[point1:point2]
        fragment2 = a2[point1:point2]
        # 交叉
        a1_1[point1:point2], a2_1[point1:point2] = a2_1[point1:point2], a1_1[point1:point2]
        # 定义容器
        a1_2 = []  # 储存修正后的head
        a2_2 = []
        a1_3 = []  # 修正后的tail
        a2_3 = []
        # 子代1头部修正
        for i in a1_1[:point1]:
            while i in fragment2:
                i = fragment1[fragment2.index(i)]
            a1_2.append(i)
        # 子代2尾部修正
        for i in a1_1[point2:]:
            while i in fragment2:
                i = fragment1[fragment2.index(i)]
            a1_3.append(i)
        # 子代2头部修订
        for i in a2_1[:point1]:
            while i in fragment1:
                i = fragment2[fragment1.index(i)]
            a2_2.append(i)
        # 子代2尾部修订
        for i in a2_1[point2:]:
            while i in fragment1:
                i = fragment2[fragment1.index(i)]
            a2_3.append(i)

        child1 = a1_2 + fragment2 + a1_3
        child2 = a2_2 + fragment1 + a2_3
        # print('修正后的子代为:\n{}\n{}'.format(child1, child2))
        return np.array(child1), np.array(child2)


class Recsbx(Recombination):
    """
    Recsbx - class : 一个用于调用内核中的函数recsbx(模拟二进制交叉)的类，
                     该类的各成员属性与内核中的对应函数的同名参数含义一致，
                     可利用help(recsbx)查看各参数的详细含义及用法。
    """

    def __init__(self, XOVR=0.7, Half_N=False, n=20, Parallel=False):
        self.XOVR = XOVR  # 发生交叉的概率
        self.Half_N = Half_N  # 该参数是旧版的输入参数Half的升级版，用于控制交配方式
        self.n = n  # 分布指数，必须不小于0
        self.Parallel = Parallel  # 表示是否采用并行计算，缺省时默认为False

    def do(self, OldChrom):  # 执行内核函数
        return self.myrecsbx(OldChrom, self.XOVR, self.Half_N, self.n, self.Parallel)

    def myrecsbx(self, OldChrom, XOVR=0.7, Half_N=False, n=20, Parallel=None):
        if Half_N is False:  # Half_N = False时返回的NewChrom的行数与OldChrom一致,当Half_N为False时配对的两条染色体相互交叉返回两条染色体。
            if (type(OldChrom) is list):
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        prob = random.random()
                        if (prob <= XOVR):
                            chrom[i], chrom[i + num // 2] = self.sbx(chrom[i], chrom[i + num // 2])
                return OldChrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    prob = random.random()
                    if (prob <= XOVR):
                        chrom[i], chrom[i + num // 2] = self.sbx(chrom[i], chrom[i + num // 2])
                return OldChrom
        elif Half_N is True:
            if (type(OldChrom) is list):
                newchrom = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    for i in range(num // 2):
                        prob = random.random()
                        if (prob <= XOVR):
                            chrom[i], _ = self.sbx(chrom[i], chrom[i + num // 2], n)
                    newchrom.append(chrom[:(num // 2)])
                return newchrom
            else:
                chrom = OldChrom
                num, len = chrom.shape
                for i in range(num // 2):
                    prob = random.random()
                    if (prob <= XOVR):
                        chrom[i], _ = self.sbx(chrom[i], chrom[i + num // 2], n)
                return OldChrom[:(num // 2)]
        else:
            if (type(OldChrom) is list):
                returnchroms = []
                for chrom in OldChrom:
                    num, len = chrom.shape
                    newchroms = []
                    for i in range(min(Half_N, num)):
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                        while point1 != point2:
                            point1 = random.randint(0, num - 1)
                            point2 = random.randint(0, num - 1)
                        nextchrom, _ = self.sbx(chrom[point1], chrom[point2], n)
                        newchroms.append(nextchrom)
                    newchroms = np.array(newchroms)
                    returnchroms.append(newchroms)
                return returnchroms
            else:
                chrom = OldChrom
                num, len = chrom.shape
                newchroms = []
                for i in range(min(Half_N, num)):
                    point1 = random.randint(0, num - 1)
                    point2 = random.randint(0, num - 1)
                    while point1 != point2:
                        point1 = random.randint(0, num - 1)
                        point2 = random.randint(0, num - 1)
                    nextchrom, _ = self.sbx(chrom[point1], chrom[point2], n)
                    newchroms.append(nextchrom)
                newchroms = np.array(newchroms)
                return newchroms

    def sbx(self, chroma, chromb, n=20):
        # print(chroma.shape)
        if len(chroma.shape) > 1:
            chroma = chroma.reshape(-1)
            chromb = chromb.reshape(-1)
        lens = chroma.shape[0]
        for k in range(lens):
            r = random.random()
            if (r <= 0.5):
                beta = np.power(2 * r, 1 / (n + 1))
            else:
                beta = np.power(1 / (2 - 2 * r), 1 / (n + 1))
            chroma[k] = 0.5 * ((1 + beta) * chroma[k] + (1 - beta) * chromb[k])
            chromb[k] = 0.5 * ((1 - beta) * chroma[k] + (1 + beta) * chromb[k])

        return chroma, chromb

    def getHelp(self):  # 查看内核中的重组算子的API文档
        help(geatpy.recsbx)


if __name__ == '__main__':
    # 测试、
    x = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    newx = copy.deepcopy(x)
    # newx[0:9] = x[9:0:-1]
    print(x, newx, x[9:0:-1], newx[0:9])
    a = []
    b = np.zeros((1, 10))
    print(type(a) is list, type(b) is np.ndarray, b.shape[1])
    dim = 2
    Encoding = "BG"
    problem_varTypes = np.array([0, 0])
    problem_ranges = np.array([[0, 0], [3, 3]])
    problem_borders = np.zeros((2, dim))
    print(crtfld(Encoding, problem_varTypes, problem_ranges, problem_borders))
    print((Encoding, problem_varTypes, problem_ranges, problem_borders))
    problem_varTypes = np.array([0, 0])
    problem_ranges = np.array([[0, 0], [3, 3]])
    problem_borders = np.zeros((2, dim))
    print(geatpy.crtfld(Encoding, problem_varTypes, problem_ranges, problem_borders))
    half = 10
    if half is True:
        print(1)
    elif half is False:
        print(0)
    else:
        print(-1)
    test = Xovdp(mutbga)
    chrom = np.random.random((4, 4))
    one = np.ones((5, 31))
    zero = np.zeros((5, 5))
    one = one.astype(np.float64)
    print(one.dtype, one[2, 1] == 1)
    testmut = Mutbin()
    newchrom = testmut.mymutbin("BG", one, Pm=1)
    newchrom1 = testmut.mymutbin("BG", zero, Pm=1)
    # print(newchrom)
    # print(newchrom1)
    test2 = Xovpmx()

    chrom = np.arange(1, 10)
    chrom = np.random.permutation(chrom)
    chroms = np.array([np.random.permutation(chrom) for i in range(4)])
    newchrom = test2.myxovpmx(chroms, XOVR=1)
    # print(chroms)
    # print(newchrom)
    random.seed(10)
    chrom = np.arange(0, 10)
    chrom1 = np.random.permutation(chrom)
    chrom2 = np.random.permutation(chrom)
    test3 = Mutinv()
    newchrom = test3.mymutinv("P", chrom1, FieldDR=None, Pm=1, InvertLen=5)
    print(chrom1)
    print(newchrom)
    zeros = np.zeros(chrom2.shape)
    # print(zeros)
    test = Recsbx()
    one = np.random.random((2, 4))
    print(chrom < 5)
    print("zuixiao", sum(np.reshape(chrom < 5, (-1))))
    one1 = copy.deepcopy(one)
    # print(one,"\n")
    newtone = test.myrecsbx(one, XOVR=1)
    print(newtone)
    # chrom = np.random.random((4, 4))
    origin = geatpy.Recsbx(Half_N=False, XOVR=0.5)
    print(origin.do(one1))
    test = Xovud()
    one = np.ones((5, 5))
    zero = np.zeros((5, 5))

    chrom = np.vstack((one, zero))  # 垂直拼接
    print(chrom)
    newchrom = test.myxovud(chrom, XOVR=0.5)
    test2 = geatpy.Xovud(XOVR=1)
    print(newchrom, "\n", test2.do(np.vstack((one, zero))))

    chrom = np.vstack((one, zero))  # 垂直拼接
    test = Mutpolyn()
    print(chrom)
    chrom = np.array(5).reshape((-1, 1))
    chrom = chrom.astype(np.float64)
    felids = np.array([0, 10, 0]).reshape((-1, 1))
    newchrom = test.mymutpolyn(Encoding="RI", OldChrom=chrom, FieldDR=felids)
    print("1", newchrom)
    test2 = geatpy.Mutpolyn()
    print(newchrom, "\n", test2.do(Encoding="RI", OldChrom=chrom, FieldDR=felids))
    # name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
    # M = 1  # 初始化M（目标维数）
    # maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
    # Dim = 1  # 初始化Dim（决策变量维数）
    # varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
    # lb = [-1]  # 决策变量下界
    # ub = [2]  # 决策变量上界
    # lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
    # ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
    # Encoding = "BG"
    # one = np.ones((5, 31))
    # Field = crtfld(Encoding, np.array(varTypes),np.stack([lb, ub], axis=0) , np.stack([lbin, ubin], axis=0))
    # print(Field)
    # print(geatpy.bs2real(one,Field))
    # help(geatpy.mutpolyn)
    # help(geatpy.crtfld)

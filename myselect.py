import numpy as np
import geatpy
import random

def binSearch(wheel, num):
    mid = len(wheel)//2
    low, high, answer = wheel[mid]
    # print( low, high,num)
    if low<=num<=high:
        return answer
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)
def susselect(select_list,selectnum):
    select_list=select_list.reshape(-1)
    # print(select_list.shape)
    N=selectnum
    wheel = []
    total = sum(select_list)
    top = 0.0
    for p in range(select_list.shape[0]):
        f = select_list[p]/total
        wheel.append((top, top+f, p))
        top += f
    # print(wheel)
    stepSize = 1.0/N
    answer = []
    r = random.random()
    answer.append(binSearch(wheel, r))
    while len(answer) < N:
        r += stepSize
        if r>1:
            r %= 1
        answer.append(binSearch(wheel, r))
    return answer
def roulette(select_list):
    sum_val = np.sum(select_list)
    random_val = random.random()
    probability = 0#累计概率
    for i in range(select_list.shape[0]):
        probability += select_list[i] / sum_val#加上该个体的选中概率
        if probability >= random_val:
            return i#返回被选中的下标
        else:
            continue
def tour(select_list,tourn):
    len=select_list.shape[0]
    choice=np.random.choice(len, tourn, replace=False)
    choose=select_list[choice]
    choosed=np.argmax(choose)
    return choice[choosed]
def select(SEL_F,FitnV_N, NSel=None, params2=None, Parallel=False):
    take = []
    if SEL_F=="rws":
        if type(FitnV_N) is int and NSel>1:
            one=np.ones((NSel,1))
            select_list=one*FitnV_N
        else:
            select_list=FitnV_N
        if NSel is not None :
            if NSel<1:

                for i in range(int(select_list.shape[0]*NSel)):
                    re = roulette(select_list)
                    # select_list[re] -= 1#被选中的下标的值减1
                    take.append(re)
                return np.array(take)
            elif NSel>1:

                for i in range(NSel):
                    re = roulette(select_list)
                    # select_list[re] -= 1#被选中的下标的值减1
                    take.append(re)
                return np.array(take)
        else:
            for i in range(int(select_list.shape[0])):
                re = roulette(select_list)
                # select_list[re] -= 1#被选中的下标的值减1
                take.append(re)
            return np.array(take)
    if SEL_F == "tour":

        if type(FitnV_N) is int and NSel>1:
            one=np.ones((NSel,1))
            select_list=one*FitnV_N
        else:
            select_list=FitnV_N

        tourn = 2   # 锦标赛参数

        if NSel is not None :
            if NSel<1:#选nsel*100%个个体

                for i in range(int(select_list.shape[0]*NSel)):
                    re = tour(select_list,tourn)
                    # select_list[re] -= 1#被选中的下标的值减1
                    take.append(re)
                return np.array(take)
            elif NSel>1:#选nsel个个体

                for i in range(NSel):
                    re = tour(select_list, tourn)
                    # select_list[re] -= 1#被选中的下标的值减1
                    take.append(re)
                return np.array(take)
        else:
            for i in range(int(select_list.shape[0])):
                re = tour(select_list, tourn)
                # select_list[re] -= 1#被选中的下标的值减1
                take.append(re)
            return np.array(take)
    if SEL_F == "ecs":

        if type(FitnV_N) is int and NSel>1:
            one=np.ones((NSel,1))
            select_list=one*FitnV_N
        else:
            select_list=FitnV_N

        EliteCopyN = select_list.shape[0]//5 if select_list.shape[0]//5>1 else 2  # 精英保留数目
        Elict = np.argsort(-select_list.reshape(-1))
        # print(Elict,select_list)
        Elicts = np.array([Elict[0]]*(EliteCopyN//2)+[Elict[1]]*(EliteCopyN//2))
        tourn = select_list.shape[0] // 2

        if NSel is not None :
            if NSel<1:#选nsel*100%个个体
                for i in range(int(select_list.shape[0]*NSel)-EliteCopyN):

                    re = tour(select_list,tourn)
                    # select_list[re] -= 1#被选中的下标的值减1
                    take.append(re)
            elif NSel>1:#选nsel个个体
                for i in range(NSel-EliteCopyN):
                    re = tour(select_list,tourn)
                    # select_list[re] -= 1#被选中的下标的值减1
                    take.append(re)
        else:
            for i in range(int(select_list.shape[0])-EliteCopyN):
                re = tour(select_list,tourn)
                # select_list[re] -= 1#被选中的下标的值减1
                take.append(re)
        # print(Elicts,np.array(take))
        return np.append(Elicts,np.array(take))
    if SEL_F == "sus":
        if type(FitnV_N) is int and NSel > 1:
            one = np.ones((NSel, 1))
            select_list = one * FitnV_N
        else:
            select_list = FitnV_N
        if NSel is not None:
            if NSel < 1:
                tmptake=susselect(select_list,int(select_list.shape[0] * NSel))
                return np.array(tmptake)
            elif NSel > 1:
                tmptake = susselect(select_list, NSel)
                return np.array(tmptake)
        else:
            tmptake = susselect(select_list, select_list.shape[0])
            return np.array(tmptake)
    if SEL_F == "dup":
        select_list=np.argsort(-FitnV_N.reshape(-1))
        if NSel<select_list.shape[0]:
            return select_list[:NSel]
        else:
            return  select_list



if __name__ == '__main__':
    N=5
    FitnV_N = np.array([i for i in range(10)]).reshape(-1,1)
    print(FitnV_N.shape[0])

    take=select("rws",FitnV_N,N)
    print("rws",take,geatpy.selecting('rws', FitnV_N,N))
    take=select("tour",FitnV_N,N)
    print("tour",take,geatpy.selecting('tour', FitnV_N,N))
    take=select("ecs",FitnV_N,N)
    print("ecs",take,geatpy.selecting('ecs', FitnV_N,N))
    take=select("sus",FitnV_N,N)
    print("sus",take,geatpy.selecting('sus', FitnV_N,N))
    take=select("dup",FitnV_N,N)
    print("dup",take,geatpy.selecting('dup', FitnV_N,N))
    choice = np.random.choice(10, 10, replace=False)
    Elicts = [1] * 3+[2]*2
    # print(Elicts)
    help(geatpy.dup)
    # print(choice)
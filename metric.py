import numpy as np


def get_hyperVolume(solutions, refer_point=np.array([1.2, 1.2])):
    '''
    计算超体积值
    :param solutions list 非支配解集，是解的目标函数值列表，形式如：[[object1,object2],[object1,object2],...]
    :param refer_point list 参考点，要被solutions内所有点支配，默认为[1.2,1.2]
    '''
    solutions = solutions[solutions[:, 0].argsort(kind="mergesort")[::-1]]  # 按第一个目标降序排序
    volume = 0
    for i in solutions:
        volume = volume + abs(i[0] - refer_point[0]) * abs(i[1] - refer_point[1])
        refer_point[0] = refer_point[0] - refer_point[0] - i[0]
    return volume


def get_igd(pf, points):
    '''
    计算IGD
    :param pf: 真实前沿点
    :param points: 计算所得前沿点
    :return:
    '''
    igd_sum = 0.0
    for p in pf:
        igd_min = float('inf')
        for pi in points:
            d = sum(list((np.array(p) - np.array(pi)) ** 2))
            if d < igd_min:
                igd_min = d
        igd_sum += igd_min
    return float(igd_sum) / len(pf)


if __name__ == '__main__':
    # Example:
    referencePoint = np.array([5, 5])
    solutions = np.array([[1, 4], [2, 2], [1, 3], [4, 1]])
    volume = get_hyperVolume(solutions)
    igds = get_igd(referencePoint, solutions)
    print(volume, igds)

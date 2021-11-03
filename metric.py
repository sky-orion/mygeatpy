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
            d = np.linalg.norm(pi - p)
            if d < igd_min:
                igd_min = d
        igd_sum += igd_min
    return float(igd_sum) / len(pf)

def get_gd(points, pf):
    '''
    计算GD
    :param pf: 真实前沿点
    :param points: 计算所得前沿点
    :return:
    '''
    igd_sum = 0.0
    for p in pf:
        igd_min = float('inf')
        for pi in points:
            d = np.linalg.norm(pi - p)
            if d < igd_min:
                igd_min = d
        igd_sum += igd_min
    return float(igd_sum) / len(pf)

def get_spacing(points):
    di = []
    lenp = points.shape[0]
    for i in range(lenp):
        spacing_min = float('inf')
        for j in range(lenp):
            if i == j:
                continue
            d = np.linalg.norm(points[i] - points[j])
            if d < spacing_min:
                spacing_min = d
        di.append(spacing_min)
    di = np.array(di)
    return np.sqrt(1 / (lenp - 1)) * np.linalg.norm(di - np.mean(di))


if __name__ == '__main__':
    # Example:
    referencePoint = np.array([5, 5])
    solutions = np.array([[1, 4], [2, 2], [1, 3], [4, 1]])
    volume = get_hyperVolume(solutions)
    igds = get_igd(referencePoint, solutions)
    from pymoo.factory import get_performance_indicator
    hv = get_performance_indicator("gd", pf=referencePoint)
    print("gd", hv.do(solutions))
    hv = get_performance_indicator("igd", pf=referencePoint)
    print("igd", hv.do(solutions))
    print("gd复现", get_gd(referencePoint, solutions))
    print("igd复现", igds)
    print(get_spacing(solutions))
    p = r'C:\Users\13927\Desktop\毕设\moea_demo\moea_demo5\Result\Objv.csv'
    with open(p, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=",")
        print(get_spacing(data))



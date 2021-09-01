import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def euler_distance(point1: list, avg: list) -> float:
    """
      计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, avg):
        distance += math.pow(a - b, 2)
    result = math.sqrt(distance)
    return round(result, 1)


# 读入数据，并进行预处理，标准的数据
def LoadDataSet(file):
    IrisDataset = pd.read_csv(file)
    feature = list(IrisDataset)
    del IrisDataset[feature[4]]  # 删除最后一列元素
    # standardscaler = StandardScaler()  # 标准化数据
    # filledDataset = pd.DataFrame(IrisDataset)
    # datasets = standardscaler.fit_transform(filledDataset.values)

    return IrisDataset  # 返回二维数组


def get_c1(all_data_list, avg):
    """
    step1: 并计算整个数据集的平均值x ；计算距平均值距离最远点为 c1
    """
    lis = list()
    map = {}
    for va in all_data_list:
        dist = abs(va - avg)
        lis.append(dist)
        map[dist] = va
    max_list = map.get(np.max(lis))
    # 离平均值距离最远的数c1

    return max_list


def add_surplus_data(c_value, data_value, vas):
    boolean = False
    dis_c1 = abs(np.mean(vas) - c_value)
    if (dis_c1 <= data_value):
        boolean = True

    return boolean


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 欧氏距离


def step2(c_list, data_list, avg):
    """
    c_value --> 对象的值
    data_list --> 所有数据集
    avg --> 均值
    step2和step3，
        step2: 将不大于 data_value的数
        step3: 选择 距平均值 最近的对象 作为更新的 聚类中心
    """
    # c1 到 avg 所有数据的平均值 的距离 除以2,记作 lambda
    data_value = distEclud(c_list, avg) / 2
    lambda_data_object_list = list()
    surplus = list()
    # 距 c1 距离不大于 lambda 的所有数据对象
    for vas in data_list:

        is_cont = distEclud(vas, c_list)
        if is_cont <= data_value:
            lambda_data_object_list.append(vas)
        else:
            surplus.append(vas)

    if len(lambda_data_object_list) == 1 or not any(lambda_data_object_list):
        return lambda_data_object_list, surplus, lambda_data_object_list
    """步骤3："""
    # 计算lambda数据对象中的平均值
    lambda_data_avg = np.average(lambda_data_object_list, axis=0)
    # 从 lambda_data_object_list 中获取离平均值最近的数据
    min_lambda_update_data_list = list()
    map_lambda_data = {}
    for val in lambda_data_object_list:  # lambda_data_object_list 是距 c1 距离不大于 lambda 的所有数据对象
        lambda_distance = distEclud(val, lambda_data_avg)
        if any(lambda_distance):
            min_lambda_update_data_list.append(lambda_distance)
            map_lambda_data[lambda_distance] = val
    if not any(min_lambda_update_data_list):
        print("======min_lambda_update_data_list为空==========")
        return lambda_data_object_list, surplus, lambda_data_object_list
    min_lambda_data_c1 = min(min_lambda_update_data_list)
    least_num_arr = map_lambda_data.get(min_lambda_data_c1)
    return least_num_arr, surplus, lambda_data_object_list


def max_distance(data, data_avg):
    """
    从 lambda_data_object_list 中获取离平均值最近的数据
    """
    distance_list = list()
    map_data = {}
    for val in data:  # lambda_data_object_list 是距 c1 距离不大于 lambda 的所有数据对象
        data_distance = abs(val - data_avg)
        distance_list.append(data_distance)
        map_data[data_distance] = val
    max_data = max(distance_list)
    surplus_max_num = map_data.get(max_data)
    return surplus_max_num


def get_cluster_central_point(c_value, data, avg):
    while True:
        new_c_value, sulplus, lambda_data = step2(c_value, data, avg)
        if ((c_value == new_c_value).all() or (not any(np.array(sulplus).tolist()))):
            return new_c_value, sulplus, lambda_data
        else:
            return get_cluster_central_point(new_c_value, data, avg)


def show(dataSet, clusterAssment, centroids):
    """
    # 画图
    """
    count = 0
    colors = ['r', 'g', 'c', 'b']
    cantCount = 0
    for point in centroids:
        color = colors[cantCount]
        plt.scatter(point[0, 0], point[0, 1], color=color, marker='x', alpha=1)
        cantCount += 1
    for point in dataSet:
        index = clusterAssment[count, 0]
        color = colors[int(index)]
        count += 1
        plt.scatter(point[0, 0], point[0, 1], color=color, marker='o', alpha=0.5)
    plt.show()


def Euler_distance(dataset, inX):
    '''

    :param dataset:
    :param inX:
    :return:
    '''
    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataset  # 计算对应元素的差值
    sqDiffMat = diffMat ** 2  # 每个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)  # 对每行元素求和
    distances = sqDistances ** 0.5  # 开根号，求得每个数据到总体均值的距离
    sortedDistIndicies = distances.argsort()
    return distances


def trance_to_one(cx0):
    """
    cx0 --> 二维的数组
    将二维数组转为一维数组
    """
    cx00 = []
    for i in cx0:
        for j in i:
            cx00.append(j)
    return cx00


'''-------------------------------------------------/
|                 step1 获取最远的点c1                 |
==================================================='''


def step1(n, c1_list, data, all_data_avg):
    '''
    获取最远点
    :param n: 行数
    :param c1_list:
    :param data:
    :param all_data_avg:
    :return:
    '''
    for i in range(n):
        c1_list[i] = get_c1(data[:, i], all_data_avg[i])
    return c1_list


def array_is_equalse(new_c_value, lamb_data):
    """
    两个数组完全一样，并且不为空
    :param new_c_value:
    :param lamb_data:
    :return:
    """
    return len(new_c_value) == len(lamb_data) and (
            distEclud(np.array(new_c_value), np.array(lamb_data)) == 0.0) \
           and not all_is_null(new_c_value, lamb_data)


def all_is_null(new_c_value, lamb_data):
    """
    判断两个数组是否为0的数组
    :param new_c_value: 1
    :param lamb_data: 2
    :return: 如果两个数组为空则返回 True，否则返回 False
    """
    new_c = len(new_c_value)
    lamb = len(lamb_data)
    boolean = not any(new_c_value) and not any(lamb_data) and new_c == lamb == 0
    return boolean


def get_cluster(c1_list, data, all_data_avg):
    """
    遍历寻找聚类中心
    :param c1_list:
    :param data:
    :param all_data_avg:
    :return:
    """
    new_c_value, surplus, lamb_data = step2(c1_list, data, all_data_avg)

    """
     all_is_null 当没有一个数据 小于lambda的数据对象的时候
     array_is_equalse  小于lambda的数据对象只有一个的时候
    """
    if array_is_equalse(new_c_value, lamb_data) or all_is_null(new_c_value, lamb_data):
        return new_c_value, surplus, lamb_data
    # 如果 返回的 new_c_value，lamb_data 完全一样，则他们是聚类中心
    if (c1_list == new_c_value).all():
        cc_value = new_c_value
        print("获取到聚类中心：", cc_value)
    else:
        new_c_value2, surplus2, lambda_data = get_cluster_central_point(new_c_value, data, all_data_avg)
        cc_value = new_c_value2
        print("找到聚类中心:", cc_value)
        surplus = surplus2
        lam = lambda_data

    return cc_value, surplus, lam


def step4_to_step6(ci_surplus, centroids, alone_point, _map):
    """
        step7 重复步骤 step4-step6 ,直到找到 k 个聚类中心为止
    :param ci_surplus: 剩余数据对象
    :param centroids: 存储聚类中心的数组
    :param alone_point: 存储孤立点的数据的数组
    :param _map: 每一个聚类中心 映射 对应小于lambda的数据对象
    :return: centroids :聚类中心, alone_point:数据孤立点, _map:剩余数据, , cf_surplus:大于epusilong的数据对象，即剩余数据对象
    """
    cf_surplus_data = np.array(ci_surplus)
    cf_n = shape(cf_surplus_data)[1]
    cf_avg = np.average(cf_surplus_data, axis=0)
    cf_list = create_array(cf_n)
    cf_list = step1(cf_n, cf_list, cf_surplus_data, cf_avg)
    cf_value, cf_surplus, cf_lambda_obj = get_cluster(cf_list, cf_surplus_data, cf_avg)

    if not all_is_null(cf_value, cf_lambda_obj) and not array_is_equalse(cf_value, cf_lambda_obj):
        """
        cf_value, cf_lambda_obj 都不是空数组
        """
        _map[one_dimensional_array_to_str(cf_value)] = make_array(cf_lambda_obj)
        centroids.append(cf_value)
    elif array_is_equalse(cf_value, cf_lambda_obj):
        """
            小于等于lambda的数据对象只有1个时，将此数据对象归位孤立点的数据
        """
        alone_point.append(cf_value)
        return step4_to_step6(cf_surplus, centroids, alone_point, _map)

    if not any(np.array(cf_surplus).tolist()) or all_is_null(cf_value, cf_lambda_obj):
        """
        小于等于lambda的数据对象只有0个，则返回
        """
        return centroids, alone_point, _map, cf_surplus
    else:
        return step4_to_step6(cf_surplus, centroids, alone_point, _map)


def create_array(n):
    """
    自动创建一维n列的数组, 自动创建数组可参考一下url地址：
    https://blog.csdn.net/weixin_39718890/article/details/110192973?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163029268816780265478103%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163029268816780265478103&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-110192973.pc_search_all_es&utm_term=python%E5%A6%82%E4%BD%95%E8%87%AA%E5%8A%A8%E5%88%9B%E5%BB%BA%E4%B8%80%E7%BB%B4%E6%95%B0%E7%BB%84&spm=1018.2226.3001.4187
    @param n 需要生成一维数组的列数
    @return 一维n列的数组，例如: n=4, 则 返回 [0,0,0,0]

    """
    return np.zeros(n)


def one_dimensional_array_to_str(ary):
    """
    将一维数组转换为 字符串
    """
    return ','.join(str(i) for i in ary)


def make_array(ary):
    return np.array(ary)

from numpy import *
import numpy as np

from com.pp.k_means_cluster.public_code import step1, get_cluster, step4_to_step6, create_array, \
    one_dimensional_array_to_str, make_array


class get_class_cluster:
    file_path = '../data/iris.csv'
    # file_path = '../data/wine.csv'
    # data = pd.read_csv(file_path)
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # skip_header=0,表示读取的csv数据，不读取第一行作为标题
    map_data_obj = {}
    # 不变的中心点
    centroids = list()
    # datasets = LoadDataSet(file_path)
    all_data_avg = np.average(data, axis=0)
    # 求总体均值，按列求均值，输出每一列的均值
    # 保存聚类中心的数据映射的小于lambda的数据对象 map
    _map = {}

    # 剩余对象平数据

    '''-------------------------------------------------/
    |                 step1 获取最远的点c1                 |
    ==================================================='''
    # 创建一个一维数组
    n = shape(data)[1]
    c1_list = create_array(n)
    c1_list = step1(n, c1_list, data, all_data_avg)

    '''-------------------------------------------------/
    |                 step2 和 step3                     |
    ==================================================='''
    # new_c_value 与 c1已经不变了的聚类中心
    new_c_value, surplus, lamb_data = get_cluster(c1_list, data, all_data_avg)
    _map[one_dimensional_array_to_str(new_c_value)] = make_array(lamb_data)
    centroids.append(new_c_value)

    # 离初始聚类中心最远的临时聚类中心
    '''-------------------------------------------------/
    |           step4  sur_num_ci作为聚类中心             |
    ==================================================='''
    # 计算剩余对象数据的平均值
    surplus = np.array(surplus)
    sur_avg = np.average(surplus, axis=0)
    ci_n = shape(surplus)[1]
    ci_list = create_array(ci_n)
    ci_list = step1(ci_n, ci_list, surplus, sur_avg)
    '''-------------------------------------------------/
    |           step5 和 step6 new_ci 作为聚类中心          |
    ==================================================='''
    ci_value, ci_surplus, ci_lambda_obj = get_cluster(ci_list, surplus, sur_avg)
    _map[one_dimensional_array_to_str(ci_value)] = make_array(ci_lambda_obj)
    centroids.append(ci_value)

    '''-------------------------------------------------/
    |           step7 重复步骤 step4-step6               |
    |                直到找到 k 个聚类中心                 |
    ==================================================='''
    # 孤立点的数据
    alone_point = list()
    # _centroids:聚类中心, _alone:数据孤立点, _cf_surplus:剩余数据, , _cf_surplus:大于epusilong的数据对象，即剩余数据对象
    _centroids, _alone, _map_data, _cf_surplus = step4_to_step6(ci_surplus, centroids, alone_point, _map)
    #


    print("++++++++++++++++++++++++++++")


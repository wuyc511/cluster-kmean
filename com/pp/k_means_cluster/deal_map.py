from com.pp.k_means_cluster.public_code import one_dimensional_array_to_str


def deal_dir(cen, map):
    for c in cen:
        m = map.get(one_dimensional_array_to_str(c))
        print("++++++++\n",c)
        print("=============\n",m)

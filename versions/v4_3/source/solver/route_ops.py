import numpy as np

def build_edge_index_routes(batch_routes, inverse=False):
    edge_index_routes = []
    for i_batch in range(len(batch_routes)):
        froms = []
        tos = []
        routes = batch_routes[i_batch]
        for r in routes:
            r0 = [0]+r+[0]
            if inverse:
                for i_tmp in range(len(r0)-2):
                    tos.append(r0[i_tmp])
                    froms.append(r0[i_tmp+1])
            else:
                for i_tmp in range(1,len(r0)-1):
                    froms.append(r0[i_tmp])
                    tos.append(r0[i_tmp+1])
        edge_index_routes.append(np.array([froms, tos]))
    return edge_index_routes

def build_edge_index_near(distMats, n, inverse=False):
    edge_index_routes = []
    for i_batch in range(len(distMats)):
        froms = []
        tos = []
        distMat = distMats[i_batch]
        for anchor in range(len(distMat)):
            for node in np.argsort(distMat[anchor])[1:n+1]:
                if inverse:
                    tos.append(node)
                    froms.append(anchor)
                else:
                    froms.append(node)
                    tos.append(anchor)
        edge_index_routes.append(np.array([froms, tos]))
    return edge_index_routes

def build_action_index(n_nodes):
    result = []
    for i in range(1,n_nodes+1):
        for j in range(1,n_nodes+1):
            result.append((i,j,True))
    for i in range(1,n_nodes+1):
        for j in range(1,n_nodes+1):
            result.append((i,j,False))
    return result

def find_ruins(n_ruins, anchor, coefficient, neighbour, routes, ruined):
    """
    与 SISR 规则类似。区别在于:
        1. 路线中被删去部分的比例是输入项，而非通过 l_max, c_bar 等参数随机。
        2. 如果下一个 neighbour 所在的线路被删除过，该点依然可以被删除，且一样会寻找线路前后的点删除。
    以上修改使得 coefficient 和删去点的空间聚集度具有更严格的正相关性
    """
    def find_t(routes, c):
        for i in range(len(routes)):
            if c in routes[i]: return i
        return None
    
    def remove_nodes(route, c, num):
        nodes = []
        i = route.index(c)
        nodes.append(route[i])
        j = i+1
        k = i-1
        num = min(num, len(route))
        while len(nodes)<num:
            if j>=len(route):
                j=0
            if k<0:
                k = len(route)-1
            nodes.append(route[j])
            if len(nodes)<num:
                nodes.append(route[k])
            j+=1
            k-=1
        return nodes

    coefficient = max(coefficient, 1e-8)
    ruin_nodes = []
    # ruined_t_indices = set([])
    for c in neighbour[anchor]:
        if c not in ruin_nodes and c!=0:
            t = find_t(routes, c)
            # if t in ruined_t_indices: continue
            num = int(np.ceil(len(routes[t]) * coefficient))
            newly_removed = remove_nodes(routes[t], c, num)
            for node in newly_removed:
                if node not in ruin_nodes and node not in ruined and len(ruin_nodes)<n_ruins: ruin_nodes.append(node)
            # ruined_t_indices.add(t)
            if len(ruin_nodes)>=n_ruins:
                break
    ruin_nodes = ruin_nodes[:n_ruins]
#     print(ruined, ruin_nodes)
    return ruin_nodes


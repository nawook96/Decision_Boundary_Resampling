import os

from tqdm import tqdm
from copy import deepcopy
import numpy as np

def decision_boundary_resampling(normal_feature, ulcer_feature, decision_boundary_train,device,
                                add_percent):
    # 이전과 알고리즘 자체가 달라진건 전혀 없음.. 코드로는 조금 바뀐부분 있으니 잘 체크해서 보도록
    new_ulcer_feature, new_normal_feature = [],[]
    print(f'normal feature len : {len(normal_feature)}')
    with tqdm(total=len(normal_feature), desc=f'Get Refined Normal Feature', unit='img') as pbar:
        for i, feature in  enumerate(normal_feature, start=0):
            feature = feature.to(device)
            prob = decision_boundary_train(feature)
            prob = prob.cpu().detach()
            
            feature = feature.cpu().detach()

            if prob[0].item() > 0.9:
                new_normal_feature.append(feature)
            pbar.set_postfix()
            pbar.update(1)
    normal_len = len(new_normal_feature)
    print(f'refine normal : {normal_len}')

    index_i = []
    index_k1 = []
    dist_list_last = []
    index_k_temp = []

    for i, ul_feature in enumerate(ulcer_feature, start=0):
        nor_dist_list = []
        index_k2 = []
        for j, no_feature in enumerate(new_normal_feature, start=0):
            if j in index_k_temp:
                continue
            nor_dist_list.append(np.linalg.norm(no_feature.numpy()-ul_feature.numpy()))
            index_k2.append(j)
        nor_dist_zip = list(zip(nor_dist_list, index_k2))
        nor_dist_zip.sort()

        dist_list_last.append(nor_dist_zip[0][0])
        index_i.append(i)
        index_k1.append(nor_dist_zip[0][1])
        index_k_temp.append(nor_dist_zip[0][1])
        if len(index_k_temp) == normal_len:
            index_k_temp.clear()
    
    short_dist = list(zip(dist_list_last, index_i, index_k1))
    short_dist.sort()

    normal_store = []
    len_uf = len(ulcer_feature)
    for i, ul_feature in enumerate(ulcer_feature, start=0):
        if i % (len_uf / (len_uf * add_percent / 100)) != 0:
            continue
        ratio_ul = 0.5
        lowest_ratio = 0
        iter_num = 100
        best_bound = 1

        ul_index = short_dist[i][1]
        nor_index = short_dist[i][2]

        temp_ulcer_feature = ulcer_feature[ul_index].to(device)

        ulcer_prob = decision_boundary_train(temp_ulcer_feature)

        ulcer_prob = ulcer_prob.cpu().detach()

        if ulcer_prob[0].item() > 0.3:
            continue
        
        for j in range(iter_num):
            ratio_ul = j/iter_num
            temp_inmediate_feature = np.add(ulcer_feature[ul_index]*ratio_ul, new_normal_feature[nor_index]*(1-ratio_ul))
            temp_inmediate_feature = temp_inmediate_feature.to(device)
            temp_inmediate_prob = decision_boundary_train(temp_inmediate_feature)

            find_bound = abs(temp_inmediate_prob[1].item()/(temp_inmediate_prob[0].item()  \
                             + temp_inmediate_prob[1].item()) - 0.5)
            if best_bound > find_bound :
                best_bound = find_bound
                lowest_ratio = ratio_ul
        if best_bound < 0.1:
            temp_feature = np.add(ulcer_feature[ul_index]*lowest_ratio, new_normal_feature[nor_index]*(1-lowest_ratio))
            new_ulcer_feature.extend(temp_feature.unsqueeze(0))
    new_ulcer_len = len(new_ulcer_feature)
    print(f'total ulcer feature : {len(ulcer_feature)}, new ulcer feature : {len(new_ulcer_feature)}')
    new_ulcer_feature.extend(ulcer_feature)
    return new_ulcer_feature, new_ulcer_len
            


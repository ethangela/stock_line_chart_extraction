import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.spatial import procrustes
from fastdtw import fastdtw
import random
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler


'''plot-extracted-data'''
#plot the extracted chart data
for img_id in range(1,7):
    f = open(f'{img_id}.json')
    chats_list = json.load(f)


    for line_id in range(len(chats_list)):
        xs = []
        ys = []
        for pos in chats_list[line_id]:
            xs.append(pos['x'])
            ys.append(-1*pos['y'])

        plt.scatter(xs, ys) 
        plt.title(f'{img_id}_{line_id+1}')
        plt.savefig(f'{img_id}_{line_id+1}.png')  # Save the figure as a PNG file
    plt.close()
            


'''pre-process'''
def shift_process(idx_list):
    for (img_id, line_dict) in idx_list:

        k_coordinates = json.load(open(f'{img_id}.json'))[line_dict['k']-1]
        for coordinate in k_coordinates:
            coordinate['y'] *= -1
        k_leftmost = min(k_coordinates, key=lambda c: c['x'])

        g_coordinates = json.load(open(f'{img_id}.json'))[line_dict['g']-1]
        for coordinate in g_coordinates:
            coordinate['y'] *= -1
        g_leftmost = min(g_coordinates, key=lambda c: c['x'])

        r_coordinates = json.load(open(f'{img_id}.json'))[line_dict['r']-1]
        for coordinate in r_coordinates:
            coordinate['y'] *= -1
        r_leftmost = min(r_coordinates, key=lambda c: c['x'])

        leftmost = min([k_leftmost, g_leftmost, r_leftmost], key=lambda c: c['x'])
        x_shift = leftmost['x']
        y_shift = leftmost['y']
        sk_coordinates = [{'x': coordinate['x']-x_shift,'y': coordinate['y']-y_shift} for coordinate in k_coordinates]
        sg_coordinates = [{'x': coordinate['x']-x_shift,'y': coordinate['y']-y_shift} for coordinate in g_coordinates]
        sr_coordinates = [{'x': coordinate['x']-x_shift,'y': coordinate['y']-y_shift} for coordinate in r_coordinates]

        with open(f'{img_id}_sk.json', 'w') as f:
            json.dump(sk_coordinates, f)
        with open(f'{img_id}_sg.json', 'w') as f:
            json.dump(sg_coordinates, f)
        with open(f'{img_id}_sr.json', 'w') as f:
            json.dump(sr_coordinates, f)

        xs = []
        ys = []
        for pos in sk_coordinates:
            xs.append(pos['x'])
            ys.append(pos['y'])
        plt.scatter(xs, ys, s=5, c='blue', label='K-line') 

        xs = []
        ys = []
        for pos in sg_coordinates:
            xs.append(pos['x'])
            ys.append(pos['y'])
        plt.scatter(xs, ys, s=5, c='green', label='EMA') 

        xs = []
        ys = []
        for pos in sr_coordinates:
            xs.append(pos['x'])
            ys.append(pos['y'])
        plt.scatter(xs, ys, s=5, c='red', label='WMA') 
        
        plt.legend()
        plt.savefig(f'{img_id}_all.png')
        plt.close()

img_id_line_dic = [ (1,{'k':6,'g':3,'r':2}), (2,{'k':8,'g':7,'r':3}), 
    (3,{'k':4,'g':3,'r':2}), (4,{'k':6,'g':4,'r':3}), 
    (5,{'k':5,'g':6,'r':2}), (6,{'k':4,'g':5,'r':1})]
shift_process( img_id_line_dic )



'''similarity'''
def align_trends_dtw(coords1, coords2):
    series1 = [(c['x'], c['y']) for c in coords1]
    series2 = [(c['x'], c['y']) for c in coords2]
    
    distance, path = fastdtw(series1, series2, dist=lambda x, y: np.linalg.norm(np.array(x) - np.array(y)))
    
    aligned_coords1 = [series1[i] for i, _ in path]
    aligned_coords2 = [series2[j] for _, j in path]
    
    return np.array(aligned_coords1), np.array(aligned_coords2)

def compute_cosine_similarity(aligned_coords1, aligned_coords2):
    vec1 = aligned_coords1.flatten()
    vec2 = aligned_coords2.flatten()
    return 1 - cosine(vec1, vec2)

def pearson_similarity(coords1, coords2):
    y1 = np.array([c['y'] for c in coords1])
    y2 = np.array([c['y'] for c in coords2])
    corr_y, _ = pearsonr(y1, y2)
    return corr_y

def euclidean_distance(coords1, coords2):
    distances = []
    for c1, c2 in zip(coords1, coords2):
        distances.append(euclidean((c1['x'], c1['y']), (c2['x'], c2['y']))) 
    return np.mean(distances) 

def procrustes_distance(coords1, coords2):
    matrix1 = np.array([(c['x'], c['y']) for c in coords1])
    matrix2 = np.array([(c['x'], c['y']) for c in coords2])
    mtx1, mtx2, disparity = procrustes(matrix1, matrix2)
    return disparity

def range_transform(disparity_matrix):
    min_disparity = np.min(disparity_matrix)
    max_disparity = np.max(disparity_matrix)
    similarity_matrix = (disparity_matrix - min_disparity) / (max_disparity - min_disparity)
    return similarity_matrix



'''matirx producing'''
def k_matrix_build(line_type='r', match_value='1.0', function=pearson_similarity):
    
    #load data
    all_coordinates = []
    for img_id in range(1,7):
        coordinate = json.load(open(f'{img_id}_s{line_type}.json'))
        all_coordinates.append(coordinate)
    matrix_similarities = np.zeros((len(all_coordinates), len(all_coordinates))) #6,6
    
    #result producing
    for i in range(len(all_coordinates)): 
        for j in range(i, len(all_coordinates)): 
            if i == j: 
                matrix_similarities[i, j] = match_value 
            
            elif i < j: 
                k1 = all_coordinates[i]
                k2 = all_coordinates[j]
                
                #data triming
                if i == 0: #expansion for Ticker 1 k-line only 
                    exp_size = len(k2) - len(k1)
                    sub_cords_b = k1[:1049]
                    sub_cords_f = k1[1062:]
                    sub_cords = k1[1049:1062]
                    x_vals = [coord['x'] for coord in sub_cords]
                    y_vals = [coord['y'] for coord in sub_cords]
                    interpolation_func_x = interp1d(range(len(x_vals)), x_vals, kind='linear') 
                    interpolation_func_y = interp1d(range(len(y_vals)), y_vals, kind='linear') 

                    num_new_points = 1062 - 1049 + exp_size 
                    new_indices = np.linspace(0, len(sub_cords) - 1, num_new_points) 
                    new_x_vals = interpolation_func_x(new_indices) 
                    new_y_vals = interpolation_func_y(new_indices) 
                    new_cords = [{'x': float(x), 'y': float(y)} for x, y in zip(new_x_vals, new_y_vals)] 

                    k1 = sub_cords_b + new_cords + sub_cords_f 
                    
                else: 
                    max_size = max(len(k1), len(k2))
                    min_size = min(len(k1), len(k2))
                    trim_size = max_size - min_size
                    if min_size < len(k1):
                        idx_to_remove = random.sample(range(len(k1)),trim_size)
                        idx_to_remove.sort(reverse=True) 
                        for idx in idx_to_remove:
                            del k1[idx]
                    elif min_size < len(k2):
                        idx_to_remove = random.sample(range(len(k2)),trim_size)
                        idx_to_remove.sort(reverse=True) 
                        for idx in idx_to_remove:
                            del k2[idx]

                #result producing
                similarity = function(k1, k2) 
                matrix_similarities[i, j] = similarity 
                matrix_similarities[j, i] = similarity # Symmetric matrix 
            
            else:
                continue
    
    return matrix_similarities

def r_matrix_build(line_type='r', match_value='1.0', function=pearson_similarity):
    
    #load data
    all_coordinates = []
    for img_id in range(1,7):
        coordinate = json.load(open(f'{img_id}_s{line_type}.json'))
        all_coordinates.append(coordinate)
    matrix_similarities = np.zeros((len(all_coordinates), len(all_coordinates))) #6,6
    
    #result producing
    for i in range(len(all_coordinates)): 
        for j in range(i, len(all_coordinates)): 
            if i == j: 
                matrix_similarities[i, j] = match_value 
            
            elif i < j: 
                k1 = all_coordinates[i]
                k2 = all_coordinates[j]

                #data triming
                max_size = max(len(k1), len(k2))
                min_size = min(len(k1), len(k2))
                trim_size = max_size - min_size
                if min_size < len(k1):
                    idx_to_remove = random.sample(range(len(k1)),trim_size)
                    idx_to_remove.sort(reverse=True) 
                    for idx in idx_to_remove:
                        del k1[idx]
                elif min_size < len(k2):
                    idx_to_remove = random.sample(range(len(k2)),trim_size)
                    idx_to_remove.sort(reverse=True) 
                    for idx in idx_to_remove:
                        del k2[idx]

                #result producing
                similarity = function(k1, k2) 
                matrix_similarities[i, j] = similarity 
                matrix_similarities[j, i] = similarity # Symmetric matrix 
            
            else:
                continue

    return matrix_similarities
    


'''results for WMA (red line)'''
#pearson_similarity
wma_similarity_matrix = r_matrix_build(line_type='r', match_value='1.0', function=pearson_similarity)

#euclidean_distance
euclidean_distance_matrix = r_matrix_build(line_type='r', match_value='0.0', function=euclidean_distance)

#procrustes_distance
procrustes_distance_matrix = r_matrix_build(line_type='r', match_value='0.0', function=procrustes_distance)

#distance-weighted adjustment
scaler = MinMaxScaler()
normalized_euclidean = range_transform(euclidean_distance_matrix)
normalized_procrustes = range_transform(procrustes_distance_matrix)
combined_distance_matrix = (normalized_euclidean + normalized_procrustes) / 2
weight_factor = 0.5
wma_similarity_matrix = wma_similarity_matrix * (1 - weight_factor * combined_distance_matrix)
print('WMA adjusted_similarity_matrix:')
print(wma_similarity_matrix)
print('\n')



'''results for k line (blue line)'''
#pearson_similarity
kline_similarity_matrix = k_matrix_build(line_type='k', match_value='1.0', function=pearson_similarity)

#euclidean_distance
euclidean_distance_matrix = k_matrix_build(line_type='k', match_value='0.0', function=euclidean_distance)

#procrustes_distance
procrustes_distance_matrix = k_matrix_build(line_type='k', match_value='0.0', function=procrustes_distance)

#distance-weighted adjustment
scaler = MinMaxScaler()
normalized_euclidean = range_transform(euclidean_distance_matrix)
normalized_procrustes = range_transform(procrustes_distance_matrix)
combined_distance_matrix = (normalized_euclidean + normalized_procrustes) / 2
weight_factor = 0.5
kline_similarity_matrix = kline_similarity_matrix * (1 - weight_factor * combined_distance_matrix)
print('K-line adjusted_similarity_matrix:')
print(kline_similarity_matrix)
print('\n')




'''weighted average results from WMA (red line) and K-line (blue line)'''
weight_factor = 0.7
average_adjusted_similarity_matrix = weight_factor * wma_similarity_matrix + (1-weight_factor) * kline_similarity_matrix
print('averaged adjusted_similarity_matrix:')
print(average_adjusted_similarity_matrix)


import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
from collections import Counter
from collections import defaultdict
from scipy.stats import rankdata

'''extrated data pre-processing'''
def calculate_fluctuation(y_values):
    differences = np.diff(np.array(y_values))  
    fluctuation = np.sum(differences**2)
    return fluctuation

def calculate_length(y_values):
    return len(y_values)

def calculate_highest_point_rank(y_values):
    return np.max(y_values)

def calculate_moving_std(y_values, window_size=2):
    moving_std = np.std([np.std(y_values[i:i+window_size]) for i in range(len(y_values) - window_size + 1)])
    return moving_std

def slope1(line):
    # x_values = np.array([point[0] for point in line])
    y_values = np.array([point[1] for point in line])
    y_uniques = np.unique(np.diff(y_values))
    # slopes = np.diff(y_values) / np.diff(x_values)  
    if max(y_uniques) >= 9:
        return 1
    else:
        return 0

def slope2(line):
    y_values = np.array([point[1] for point in line])
    y_differ = np.diff(y_values, n=4)
    y_uniques = np.unique(y_differ)
    max_consecutive_diff = np.max(np.diff(y_uniques))
    return max_consecutive_diff
    
def segments(line):
    y_values = np.array([point[1] for point in line])
    slopes = np.diff(y_values)

    segments = defaultdict(int)
    current_slope = slopes[0]
    current_start = 0

    for i in range(1, len(slopes)):
        if slopes[i] == current_slope:
            continue
        else:
            segment_length = i - current_start
            if abs(current_slope) >= 0:
                segments[current_slope] += segment_length
            current_start = i
            current_slope = slopes[i]
    segment_length = i - current_start
    if abs(current_slope) >= 0:
        segments[current_slope] += segment_length
    
    return segments #np.unique(np.array(segments))

def highest_point(line):
    x_values = np.array([point[0] for point in line])
    y_values = np.array([point[1] for point in line])
    idx = np.argmax(y_values)
    return x_values[idx], np.max(y_values)


def slopes(line,shift=10):
    y_values = np.array([point[1] for point in line])
    y_differ = y_values[shift:] - y_values[:-shift]
    y_uniques = np.unique(y_differ)
    return y_uniques


def invert_y_coordinates(coords):
    return [[point["x"], -1*point["y"]] for point in coords]

for img_id in range(1,7):
    f = open(f'{img_id}.json')
    chats_list = json.load(f)
    smooth = []
    
    print(f'{img_id}')
    m = [0] * len(chats_list)
    xs, hs = [], []
    for line_id in range(len(chats_list)):
        coords = invert_y_coordinates(chats_list[line_id])
        if slope1(coords):
            m[line_id] += 1
        if slope2(coords) > 8:
            m[line_id] += 1
        x, h = highest_point(coords)
        xs.append(x)
        hs.append(h)
        # print(f'{line_id+1}', slopes(coords))
    
    x_idx = xs[np.argmax(hs)]
    ys = []
    for line_id in range(len(chats_list)):
        coords = invert_y_coordinates(chats_list[line_id])
        for i, pair in enumerate(coords):
            if pair[0] == x_idx:
                ys.append(pair[1])
            # if len(ys) != i+1:
            #     for j in range(10):
            #         if pair[0]+j == x_idx:
            #             ys.append(pair[1]+j)
            #             break

                
    threshold = 2
    sorted_ys = sorted(set(ys), reverse=True)
    ranks = {}
    rank = 1
    previous_value = sorted_ys[0]
    for v in sorted_ys:
        if abs(previous_value - v) > threshold:
            rank += 1
        ranks[v] = rank
        previous_value = v
    ys_rank = [ranks[v] for v in ys]
    ys_rank_adjust = [x if m[i]==0 else 1000 for i,x in enumerate(ys_rank)]

    final_rank = rankdata(ys_rank_adjust, method='dense').astype(int).tolist()


    # print(m)
    # print(ys_rank_adjust)
    print(final_rank)
    print('\n')
    print('\n')
    # print(smooth)
    

            

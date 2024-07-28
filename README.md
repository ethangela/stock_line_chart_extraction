# Assignment 2

## Introduction

Data extraction from line-chart images is an essential component of the automated document understanding process, this project focuses on first extracting line chart data from the raw screenshot image, and then exploring the similarites among the extrated data.

## Workflow

### 1. Extract line chart data from image
We utilize techniques from the ICDAR 2023 paper, `LineFormer - Rethinking Chart Data Extraction as Instance Segmentation`, to extract line data through instance segmentation. The code is adapted from [LineFormer](https://github.com/TheJaeLal/LineFormer), and we use their pre-trained model directly for this task.

### 2. Extract the useful information from the extracted data above

The line chart data extracted in the previous step is often messy and requires further preprocessing for downstream tasks.

For instance, although Ticker1.png contains only 4 lines, the previous step identified 6 lines due to the difficulty of meticulous accurate detection. Using the extracted coordinates and replotting these 6 lines on the original image appears as follows:
![Project Screenshot](./images/screenshot.png)
Coordinates and replots of all 6 tickers can be found at `./chart_extract_raw_output`.

Thus, we need to analyze these extracted lines to determine which 4 lines are useful. This process involves manual testing, examining each line individually. Using Ticker1.png as an example, we found that three detected lines are suitable for downstream tasks. We replot these three lines in another figure:
![Project Screenshot](./images/screenshot.png)
Although the original Ticker1.png lacks legend information, we infer that the corresponding lines should represent the k-line, EMA, and WMA. The SMA line, which seems to appear as the yellow line in the original Ticker1.png, is not detected, possibly due to its subtle yellow color.

The coordinates of these three lines have been adjusted so that the leftmost point of the entire figure is at (0,0). The complete coordinate data is saved in a JSON file and is available in the `./chart_extract_finetune_json`.

These preprocessing steps are applied to all 6 tickers.

### 3.  Handling fifferent lengths of extracted coordinates among 6 Tickers

The extracted coordinates of the k-line, EMA, and WMA lines for each ticker may vary in length, which we need to address before calculating similarity. For example, when calculating the Pearson correlation between two lines, they must have the same length. The details of the lengths of the k-line, EMA, and WMA lines for each ticker can be found in `./length.txt`.

First, we observed significant fluctuations in the lengths of the EMA lines, so we excluded EMA lines from the similarity analysis. Next, we found that, except for `k_line_1` (the extracted k-line for ticker 1), all other k-lines have a similar number of coordinates. All wma-lines also have a similar number of coordinates. We manually identified that the existence of some sparse points in `k_line_1` results such lower number of coordinates. Consdiering this, whenever dealing with `k_line_1`, we use interpolation to increase its length. An example of this interpolation is shown below:

![Project Screenshot](./images/screenshot.png)
![Project Screenshot](./images/screenshot.png)

For other lines, when comparing two lines, such as `wma_line_4` and `wma_line_6`, we find the minimum length of the two lines and randomly remove few points from the longer line to align their lengths.


### 3. Calculate pearson similarity, euclidean distance and procrustes distance of both k-lines and wma-lines of 6 tickers

We calculate three types of similarity metrics—Pearson similarity, Euclidean distance, and Procrustes distance—for both K-lines and WMA-lines of the 6 tickers.

Pearson similarity measures the linear correlation between two sets of points. It ranges from -1 to 1, where 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.

For example, the Pearson similarity matrix for WMA-lines is:

|          | Ticker 1    | Ticker 2    | Ticker 3    | Ticker 4    | Ticker 5    | Ticker 6    |
|----------|-------------|-------------|-------------|-------------|-------------|-------------|
| **Ticker 1** | 1.0000      | 0.9471      | 0.8926      | 0.3973      | 0.1124      | 0.3563      |
| **Ticker 2** | 0.9471      | 1.0000      | 0.9060      | 0.4124      | 0.1466      | 0.3736      |
| **Ticker 3** | 0.8926      | 0.9060      | 1.0000      | 0.4520      | 0.2103      | 0.4398      |
| **Ticker 4** | 0.3973      | 0.4124      | 0.4520      | 1.0000      | 0.6938      | 0.8028      |
| **Ticker 5** | 0.1124      | 0.1466      | 0.2103      | 0.6938      | 1.0000      | 0.7105      |
| **Ticker 6** | 0.3563      | 0.3736      | 0.4398      | 0.8028      | 0.7105      | 1.0000      |


Euclidean distance measures the straight-line distance between two points in Euclidean space. For two lines represented by their coordinates, the Euclidean distance quantifies the overall distance between the corresponding points on the two lines.

Procrustes distance measures the similarity between two shapes by optimally translating, rotating, and scaling one shape to best match the other.

To fine-tune the Pearson similarity matrix, we normalize the Euclidean distance and Procrustes distance matrices to be in the range of 0 to 1, and then use these distance matrices as weights:

`combined_distance_matrix = (normalized_euclidean + normalized_procrustes) / 2`

`weight_factor = 0.5`

`wma_weighted_similarity_matrix = wma_similarity_matrix * (1 - weight_factor * combined_distance_matrix)`

This weight factor value, 0.5,  is determined through initial tests and can be further fine-tuned.

Similarly, we apply the same process to the K-lines and obtain a weighted K-line similarity matrix `kline_weighted_similarity_matrix`.

Now we have two matrice, and we would like to produce a combined similarity matrix by averaging them:

`weight_factor = 0.7`

`final_similarity_matrix = weight_factor * wma_weighted_similarity_matrix + (1-weight_factor) * kline_weighted_similarity_matrix`

This factor value, 0.7,  is chosen because we observed that WMA lines are extracted more accurately than K-lines.

The final matrix is presented below:


|          | Ticker 1    | Ticker 2    | Ticker 3    | Ticker 4    | Ticker 5    | Ticker 6    |
|----------|-------------|-------------|-------------|-------------|-------------|-------------|
| **Ticker 1** | 1.0000      | 0.9318      | 0.8701      | 0.3825      | 0.0815      | 0.3307      |
| **Ticker 2** | 0.9318      | 1.0000      | 0.8903      | 0.3949      | 0.1117      | 0.3472      |
| **Ticker 3** | 0.8701      | 0.8903      | 1.0000      | 0.4540      | 0.1904      | 0.4221      |
| **Ticker 4** | 0.3825      | 0.3949      | 0.4540      | 1.0000      | 0.6763      | 0.7918      |
| **Ticker 5** | 0.0815      | 0.1117      | 0.1904      | 0.6763      | 1.0000      | 0.7001      |
| **Ticker 6** | 0.3307      | 0.3472      | 0.4221      | 0.7918      | 0.7001      | 1.0000      |


All results can be reproduce using the code `matrix_produce.py`


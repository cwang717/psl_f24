# Project 2: Walmart Store Sales Forcasting

Author: Chaojie Wang (netID: 656449601) UIUC MCS Online Fall 2024

## Technical Details

### Data Pre-processing

1. Data Loading and Cleaning:
   - We loaded the training data from CSV files.
   - Dates are converted to datetime format and decomposed into Week and Year components
   - Weeks are categorized from 1-52 for consistent handling
   - Missing values in the sales data are filled with zeros

2. Sales Data Smoothing
   - Applied SVD (Singular Value Decomposition) smoothing by department
   - Process:
     - Data is pivoted to create a store-by-date matrix
     - Store means are subtracted to center the data
     - SVD decomposition is performed
     - Data is reconstructed using reduced components (8 principal components) and store means are added back
   - This helps reduce noise while preserving important sales patterns

### Model Implementation

- Separate models are fitted for each Store-Department combination (for Store-Department pairs that do not exist in the training set, we use 0 as the prediction)
- Features include:
  - Store and Department indicators
  - Year (linear and quadratic terms)
  - Week indicators
- Multicollinearity handling:
  - Removes columns that are all zeros
  - Identifies and removes linearly dependent columns using least squares residuals
  - Uses Ordinary Least Squares (OLS) regression from statsmodels

### Holiday Period Adjustment

- Special handling for Christmas period (weeks 48-52)
- Implements a circular shift for holiday sales predictions:
  - Identifies cases where middle weeks (49-51) are 10% higher than edge weeks
  - Applies a fractional shift (2/7) of sales between adjacent weeks
- This helps account for holiday shopping pattern variations

## Performance Metrics

The code was run on a Ubuntu desktop (CPU: AMD Ryzen 9 7950X3D 16-Core Processor - 5.7GHz; Memory: 64GB). The execution times and WMAE scores for each fold were as follows:

1. Fold 1: 20.75 seconds
   - WMAE: 1943.344
2. Fold 2: 22.10 seconds
   - WMAE: 1390.886
3. Fold 3: 21.33 seconds
   - WMAE: 1392.232
4. Fold 4: 21.72 seconds
   - WMAE: 1523.191
5. Fold 5: 23.85 seconds
   - WMAE: 2252.510
6. Fold 6: 22.25 seconds
   - WMAE: 1636.621
7. Fold 7: 22.81 seconds
   - WMAE: 1615.023
8. Fold 8: 26.58 seconds
   - WMAE: 1362.546
9. Fold 9: 24.97 seconds
   - WMAE: 1350.826
10. Fold 10: 24.59 seconds
    - WMAE: 1332.109

Average WMAE across all folds: 1579.929

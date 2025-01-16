import numpy as np
import pandas as pd

def cal_gini(class_values, splits):
    '''calculate the gini index based on classes and data splits'''
    gini = 0.0
    total_size = 0
    
    for split in splits:
        size = len(split)
        if size == 0: continue
        for class_value in class_values:
            proportion = [row[-1] for row in split].count(class_value)/ float(size)
            gini += (proportion*(1.0-proportion))*size
        total_size += size

    return gini/total_size

def test_split(col_index, threshold, datasets):
    '''list all the potential splits for each col'''
    left, right = [], []
    
    for row in datasets:
        if row[col_index] < threshold: left.append(row)
        else: right.append(row)
            
    return left, right

def get_split(datasets):
    '''find the best split'''
    class_values = list(set(list([row[-1] for row in datasets])))
    best_col, best_threshold, best_score, best_group = float("inf"), float("inf"), float("inf"), None
    
    for col in range(len(datasets[0])-1):
        for row in range(len(datasets)):
            group = test_split(col, datasets[row][col], datasets)
            # use gini index to measure impurity here
            gini_index = cal_gini(class_values, group)
            if gini_index < best_score:
                best_col, best_threshold, best_score, best_group = \
                col, datasets[row][col], gini_index, group
                
    return {'col_index': best_col, 'threshold': best_threshold, 'groups': best_group}

# Create child splits for a node or make terminal
def split_tree(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split_tree(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split_tree(node['right'], max_depth, min_size, depth + 1)

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

class MyDecisionTreeClassifier:
    
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.tree = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If Gradient Boost
        self.loss = loss
        
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y):
        datasets = np.column_stack([X, y])
        root = get_split(datasets)
        split_tree(root, self.max_depth, self.min_samples_split, 1)
        return root
        
    def predict_point(self, row, node=None):
        if not node: node = self.tree
        
        if row[node["col_index"]] < node["threshold"]:
            if isinstance(node["left"], dict):
                return self.predict_point(row, node["left"])
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self.predict_point(row, node["right"])
            else:
                return node["right"]
    
    def predict(self, X_test, node=None):
        if not node: node = self.tree
        prediction = []
        
        for row in X_test:
            prediction.append(self.predict_point(row,self.tree))
        
        return np.array(prediction)
    
    
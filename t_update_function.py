import pandas as pd
import numpy as np
import time


def update_threshold(r_losses, true_labels, current_threshold):
    """
    Update the threshold using labelled data.
    
    Args:
    r_losses (numpy.ndarray): The reconstruction errors.
    true_labels (numpy.ndarray): The labels (0 for normal, 1 for anomalous).
    current_threshold (float): The current threshold.
    
    Returns:
    float: The updated threshold.
    """

    if np.sum(true_labels == 1) < 10:
        return current_threshold
    
    # Define a range of possible thresholds
    #thresholds = np.linspace(np.min(r_losses), np.max(r_losses), 1000)
    thresholds = np.linspace(current_threshold - (current_threshold*0.50), current_threshold + (current_threshold*0.50), 1000)
    
    # Initialize the best threshold and the best score
    best_threshold = current_threshold
    best_score = -np.inf
    
    # Iterate over all possible thresholds
    for threshold in thresholds:
        # Calculate true positives, false positives, true negatives, and false negatives
        tp = np.sum((r_losses >= threshold) & (true_labels == 1))
        fp = np.sum((r_losses >= threshold) & (true_labels == 0))
        tn = np.sum((r_losses < threshold) & (true_labels == 0))
        fn = np.sum((r_losses < threshold) & (true_labels == 1))
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fp) > 0 else 0
        f1_score = 2 / ( (1 / precision) + (1 / recall)) if (precision + recall) >0 else 0
        # print("Valeur du F1 : ",f1_score)
        # print("Threshold : ",threshold)
        # print("Précision : ", precision)
        # print("Recall: ", recall)
        # print("TP: ", tp)
        # print("FP: ", fp)
        # print("FN: ", fn)
        

        # Update the best threshold if the current F1 score is better
        if f1_score > best_score:
            best_score = f1_score
            best_threshold = threshold
    print("-----------------------------------------------------THRESHOLD UPDATED-----------------------------------------------")
    print("Best Threshold found : ", best_threshold)
    print("Best F1_score found : ", best_score)
    print("---------------------------------------------------------------------------------------------------------------------")

            
    return best_threshold

if __name__ == '__main__':
    loss = np.array([0.2,1,3])
    labels = np.array([0,0,1])

    tr = update_threshold(loss,labels,0.19)

    print(tr)
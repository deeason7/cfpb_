Model Training & Evaluation Summary

** Training progression: 
    * Started at 46.2% training accuracy in epoch 1 and steadily improved each epoch.
    * By epoch 10, training accuracy reached 92.95%, and by epoch 15 it peaked at 96.39%. 

** Validation performance: 
    * Validation accuracy climbed from 54.94% in epoch 1 to a high of 94.86% at epoch 11. 
    * After reducing the learning rate, val_accuracy stabilized around 94-95% through epoch 15. 

** Final test accuracy: 
   *The model achieved 94.4% accuracy on the held-out test set. 

** Per-class metrics(test set): 
   * Extreme Negative: Precision 98.1%, Recall 96.4%, F1 97.2% (2,695 samples)
   * Negative: Precision 95%, Recall 90.8%, F1 92.8% (3,943 samples)
   * Neutral: Precision 91.1%, Recall 97.0%, F1 93.9% (3,383 samples)

** Overall:
   * Macro-averaged F1: 94.7%
   * Weighted-averaged F1: 94.4%

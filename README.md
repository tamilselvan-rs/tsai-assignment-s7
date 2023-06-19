# TSAI Assignment 7

## Question 1

For your first attempt, please share your:

- Targets  
- Results (must include best train/test accuracies and total parameters)  
- Analysis  
- File Link

### Target
- 99.4% Accuracy
- No limit on number of parameters
- RF >= 20

### __Attempt 1__

### Result (Within 15 Epoch)
| Metric              | Output |
| --------------------|--------|
| Best Train Accuracy | 98.65% |
| Best Test Accuracy  | 99.32% |
| #of Parameters      | 17,332 |
| RF Out              | 26     |
| Batch Size          | 512    |
| LR                  | 0.02   |

### Analysis
- Model isn't overfitting
- Tried adjusting the LR and got 99.4% at 0.02
- Number of parameters are too high for the assignment target
- Applying Dropout 10% of after input block significantly affects accuracy (possibly due to avg pooling following it)


File Link:  
[Base Model](model-base/model.py)  
[ipynb](model-base/S7-Base.ipynb)

---

### __Attempt 2__

### Result (Within 15 Epoch)
| Metric              | Output |
| --------------------|--------|
| Best Train Accuracy | 98.74% |
| Best Test Accuracy  | 99.41% |
| #of Parameters      | 17,332 |
| RF Out              | 26     |
| Batch Size          | 128    |
| LR                  | 0.01   |

### Analysis
- Reducing batch size and adjust LR has significantly helped in improving train accuracy
- Model isn't overfitting
- Number of parameters are too high for the assignment target
- Model is beautiful primarily because the train accuracy improvements are transferred to test accuracy

File Link:  
[Batch 128](model-batch-128/model.py)  
[ipynb](model-batch-128/S7-BS-128.ipynb)

## Question 2

For your second attempt, please share your:

- Targets  
- Results (must include best train/test accuracies and total parameters)  
- Analysis  
- File Link

### Target
- No of Parameters <= 8000

### __Attempt 1__

### Result (within Epoch 15)
| Metric              | Output |
| --------------------|--------|
| Best Train Accuracy | 97.97% |
| Best Test Accuracy  | 98.75% |
| #of Parameters      | 14,490 |
| RF Out              | 30     |
| Batch Size          | 128    |
| LR                  | 0.01   |

### Analysis
- Model's accuracy has become poor (98.75% from 99.4%)
- Possibly too less convolutions and lot of pooling
- #of parameters saw a dip of 2K still too high 
- Model isn't overfitting but training accuracy has suffered

File Link:  
[Model with GAP](model-gap/model.py)  
[ipynb](model-gap/S7-GAP.ipynb)

---

### __Attempt 2__

### Result (within Epoch 15)
| Metric              | Output |
| --------------------|--------|
| Best Train Accuracy | 98.75% |
| Best Test Accuracy  | 99.37% |
| #of Parameters      | 7,992 |
| RF Out              | 28     |
| Batch Size          | 128    |
| LR                  | 0.01   |

### Analysis
- Increasing the no of convolutions and reducing the channels to save on # of parameters has resulted in better accuracy
- We have reached the accuracy of 17K Param model
- Model isn't overfitting
- Model is achieving 99.4 consistently from 16th Epoch
- Number of parameters is under allowed limit

File Link:  
[GAP with More Conv](model-gap-conv-4/model.py)  
[ipynb](model-gap-conv-4/S7-GAP-Conv4.ipynb)


## Question 3

For your third attempt, please share your:

- Targets  
- Results (must include best train/test accuracies and total parameters)  
- Analysis  
- File Link  

### Target
- 99.4% accuracy before 15th Epoch
- Push the Training Accuracy

### Result
| Metric              | Output |
| --------------------|--------|
| Best Train Accuracy | 98.85% |
| Best Test Accuracy  | 99.39% |
| #of Parameters      | 7,992  |
| RF Out              | 28     |
| Batch Size          | 128    |
| LR                  | 0.01   |

### Analysis

- Random Shift and Random Rotation added to data has improved the training accuracy & the model achieved max training & test accuracy than its previous version
- Model isn't overfitting
- Achieved 99.4 in 16th Epoch onwards

File Link:  
[Model with Data Augmentation](model-data-aug/model.py)  
[ipynb](model-data-aug/S7-GAP-Data-Aug.ipynb)


## Question 4

What is the minimum value of Dropout have you used? (Mention 0.0 if you haven't used dropout anywhere)

0.1

## Question 5

What is the Receptive field of your Fifth Model? Use the new formula we covered in class. 

28

## Question 6

Have you used Data Augmentation (like rotation)?

Yes

## Question 7

How many parameters your final model has? (this awards only those who out-performed the assignment target) 

7992
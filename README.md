## Part 1 — Results and interpretation

I built a PyTorch MLP to predict each stock’s next trading-day closing price using its current-day open, high, low, close, and volume.

### Data exploration and preparation

I explored the split-adjusted price table, which contains 851,264 observations for 501 stock symbols. I found no missing values or duplicate stock–date pairs. However, I identified 21 zero-volume observations and differences in history length between stocks.

I split the prediction examples chronologically by target date: 2010–2014 for training, 2015 for validation, and 2016 for testing. I standardized inputs and targets using parameters calculated from training data only.

### Hyperparameter tuning

I used GridSearchCV to evaluate 16 hyperparameter combinations with a predefined chronological validation split. The selected configuration used:

- Two hidden layers with 64 and 32 neurons.
- ReLU activations.
- The Adam optimizer.
- A learning rate of 0.001.
- 20 training epochs.

I kept the batch size fixed at 1,024.

### Learning curves

Training loss decreased rapidly during the first epochs. Validation loss improved overall but fluctuated, without a sustained increase.

I used MSE for the learning curves and MAE/RMSE for evaluation because ordinary classification accuracy is not appropriate for continuous price predictions. I monitored validation performance during development and reserved the test set for final evaluation.

### Test results

| Method | Test MAE (USD) | Test RMSE (USD) |
| --- | ---: | ---: |
| Current-close baseline | **0.9508** | **2.1505** |
| Initial MLP | 0.9814 | 2.1904 |
| Tuned MLP without regularization | 0.9708 | 2.1819 |
| Dropout | 2.4454 | 5.0558 |
| Weight decay | 1.0072 | 2.2652 |
| Dropout + weight decay | 1.1088 | 2.3487 |

The tuned MLP achieved a training MAE of 0.6592 USD and a test MAE of 0.9708 USD. The higher test error indicates weaker performance on unseen dates, although differences between the periods also contribute to this gap.

Hyperparameter tuning improved the initial network’s performance. However, the tuned model did not outperform the baseline that predicts the next closing price as equal to the current closing price.

### Effect of regularization

I tested dropout, weight decay, and their combination. None improved validation performance at the tested settings, and their test errors were also higher than those of the model without regularization. Dropout alone produced the largest errors.

These results apply to the regularization strengths and training duration tested. They do not establish that regularization cannot help under other settings.

### Limitations

The reported metrics aggregate stocks with different price levels and history lengths. They do not imply equal performance for every stock. Some validation and test stocks were also absent from training.

### What I learned

I learned how to prepare chronological prediction targets, standardize data without using future information, train a PyTorch network, tune hyperparameters, and compare regularization techniques.

Most importantly, I learned that decreasing training loss does not guarantee a useful model: performance must also be assessed on unseen data against a simple baseline.

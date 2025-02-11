# Original Authors:

Yunkai Zhang(yunkai_zhang@ucsb.edu) - University of California, Santa Barbara

Qiao Jiang - Brown University

Xueying Ma - Columbia University

Acknowledgement: Professor Xifeng Yan's group at UC Santa Barbara. Part of the work was done at WeWork.

https://github.com/zhykoties/TimeSeries


## To run:
1. Install all dependencies listed in requirements.txt. Note that the model has only been tested in the versions shown in the text file.

1. Download the dataset and preprocess the data:
  
   ```bash
   python preprocess.py
   ```
1. Start training:
  
   ```bash
   python train.py
   ```
   
   - If you want to perform ancestral sampling,
   
        ```bash
        python train.py --sampling
        ```
   - If you do not want to do normalization during evaluation,
              
   
        ```bash
        python train.py --relative-metrics
        ```
1. Evaluate a set of saved model weights:
        
   ```bash
   python evaluate.py
   ```
1. Perform hyperparameter search:
        
   ```bash
    python search_params.py
   ```

## Reference:
The reimplementation of the DeepAR paper(DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks https://arxiv.org/abs/1704.04110) is available in PyTorch. 


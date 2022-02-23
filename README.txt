datasets: This directory stores restaurant14/Restaurant15/Restaurant16/Laptop15/Laptop16 processed datasets
models: This directory stores our proposed PSI model
data_utils_I_P.py: This python file is the sampling processing of datasets of Interacting Polarity
data_utils_I_P_L_A.py: This python file is the sampling processing of datasets of Interacting Polarity and Limiting Aspect
data_utils_I_A.py: This python file is the sampling processing of datasets of Interacting Aspect
data_utils_I_A_L_P.py: This python file is the sampling processing of datasets of Interacting Aspect and Limiting Polarity
train.py: This python file is the training function of PSI model 

To run train.py, you can input following command:
python train.py --dataset res14 --lr 2e-5 --n_classes 3 --n_samples 4

'--dataset' parameter can be set to res15/res16/laptop15/laptop16 datasets
'--lr' parameter means learning rate, it can be set to 5e-5, 2e-5
'--n_classes' parameter means the number of classes, it can be set to 2, 3 
'--n_samples' parameter means the number of samples per class, it can be set to 3, 4, 5 


# How Will My Business Process Unfold? Predicting Case Suffixes With Start and End Timestamps
This project contains supplementary material for the article **"How Will My Business Process Unfold? Predicting Case Suffixes With Start and End Timestamps"** 


Our project builds upon an existing BiLSTM-based approach for next activity and case suffix prediction. We have expanded this initial implementation to include sweep line based approach, alongside various baseline  approaches that are detailed in our paper. 

# Datasets:
Datasets can be found in the input folder 

# Reproduce Results:

* To execute this code you need to install anaconda in your system, and create an environment using environment.yml specification provided in the repository.
  ```
  cd GenerativeLSTM
  conda env create -f environment.yml
  conda activate lstm-caise23
  ```
## Running the script

Once created the environment, you can perform training and testing by running the following python scripts dg_training.py  and dg_predictiction.py  through command line as  described below:

*Training LSTM for suffix prediction:* To perform this task you need to set the required activity (-a) as 'training' followed by the name of the (-f) event log, and all the following parameters:

* Filename (-f): Log filename.
* Model family (-m): lstm
* Max Eval (-e): Maximum number of evaluations.
* Opt method (-o): Optimization method used. The available options are hpc and bayesian.

```
(lstm_env) C:\sc_lstm>python dg_training.py -f BPI_Challenge_2012.csv -m lstm -e 1 -o bayesian
```
* For training purposes, event log should exist in the input folder
* For training LSTM on different architectures, with in the dg_training.py file please specify the name of the desired architecture to train a model.


*Predictive task:* It is possible to predict case suffix.  To perform this task, you need to set the activity (-a) as ‘pred_sfx’ for case suffix. Additionally, it's required to indicate the folder where the predictive model is located (-c), and the name of the .h5 model (-b). Finally, you need to specify the method for selecting the next predicted task (-v) ‘random_choice’ or ‘arg_max’ or any other sampling approach and the number of repetitions of the experiment (-r).

```
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "arg_max" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "topk" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "d_action" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "random_choice" -r 1
(lstm_env) C:\sc_lstm>python ./dg_predictiction.py -a pred_sfx -c BPI_Challenge_2012 -b "BPI_Challenge_2012.h5" -v "nucleus" -r 1


```

* The trained predictive models for the next activity, start and duration will be saved in the output folder along with the parameters and test set.
* After running the test set on the trained model, results for sweep line based approach can be found in the same folder


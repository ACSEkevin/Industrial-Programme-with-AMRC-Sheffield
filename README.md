# Industrial-Training-Programme-with-AMRC-Sheffield
### ITP Additive Manufacturing (Process Monitoring) MA3<p>
Please note that the dataset has been removed from `data/` directory in case of dataset leakage, remember to add the dataset in this directory and modify the data path in `data_processing.py` before running the code<p>
The models in the project are developed using Keras/TensorFlow (MLP) and Scikit-Learn (AdaBoost, XGBoost, LightGBM, same API), any questions please refer to 
  1. [Keras tutorial: build a model in class object](<https://keras.io/guides/making_new_layers_and_models_via_subclassing/>)
  2. [TensorFlow tutorial: model save checkpoint, weights saving and loading](<https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn>)
  3. [Sklearn ensemble tutorial: Ensemble learning  & AdaBoost](<https://scikit-learn.org/stable/modules/ensemble.html>)
  4. [Numpy quick start](<https://numpy.org/doc/stable/user/quickstart.html>)
  5. [XGBoost sklearn API tutorial](<https://xgboost.readthedocs.io/en/stable/python/python_api.html>)
  6. [LightGBM sklearn API tutorial](<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>)
  
## Directory Description<p>
### ITP_MA3_ProcessMonitoring/<p>
<b>`checkpoint/`</b>: storing weightsHDF5 file<br>
<b>`data/`</b>: storing dataset csv file<br>
<b>`itpma3_utils/`</b>: 
1. `utils.py`: wrapping functions and classes that are frequently used
2. <b>`models/`</b>: machine learning models<br>

`data_processing.py`: data analysis, preprocessing, feature engineering<br>
`train.py`: model training<br>
`evaluate.py`: model evaluation<br>
`requirements.py`: for version test and available packages detecting

## Author pages<p>
The project has six contributors. All the page links will be refined in the future<p>
#### @ACSEKevin <https://github.com/ACSEkevin><br>

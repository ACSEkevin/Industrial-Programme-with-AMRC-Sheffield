# Industrial-Training-Programme-with-AMRC-Sheffield
### ITP Additive Manufacturing (Process Monitoring) MA3<p>

Please note that the dataset has been removed from `data/` directory in case of dataset leakage, remember to add the dataset in this directory and modify the data path in `data_processing.py` before running the code<p>

To clone the project:

```
git clone https://github.com/ACSEkevin/Industrial-Programme-with-AMRC-Sheffield.git
```

## Directory Description<p>
### ITP_MA3_ProcessMonitoring/<p>
<b>`checkpoint/`</b>: storing weightsHDF5 file<br>
<b>`data/`</b>: storing dataset csv file<br>
<b>`itpma3_utils/`</b>: 
* `utils.py`: wrapping functions and classes that are frequently used
* <b>`models/`</b>: machine learning models<br>

`data_processing.py`: data analysis, preprocessing, feature engineering<br>
`train.py`: model training<br>
`evaluate.py`: model evaluation<br>
`requirements.py`: for version test and available packages detecting
 
The notebook version of data processing, model training and evalutaion are also provided:<br>
[![processing](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EbNVMcqn_QE8-m6F6W2hEqNMnQkGYBR6)
[![train](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13YQa9RBM95bxBGivlZyJrVsuRGVVFoqm)
[![evaluate](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e3V0FFxW_jyptdssxj3_SzVZzLV6t_Ps)
 
<b>NOTICE:</b>
 Please change the directory before run the code, in colab, this command will help with that:<br>
 
 
```python
from google.colab import drive
drive.mount('/content/drive') 
```

 
## Tutorials<p>
The models in the project are developed using Keras/TensorFlow (MLP) and Scikit-Learn (AdaBoost, XGBoost, LightGBM, same API), any questions please refer to 
  * [Keras tutorial: build a model in class object](<https://keras.io/guides/making_new_layers_and_models_via_subclassing/>)
  * [TensorFlow tutorial: model save checkpoint, weights saving and loading](<https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn>)
  * [Sklearn ensemble tutorial: Ensemble learning  & AdaBoost](<https://scikit-learn.org/stable/modules/ensemble.html>)
  * [Numpy quick start](<https://numpy.org/doc/stable/user/quickstart.html>)
  * [XGBoost sklearn API tutorial](<https://xgboost.readthedocs.io/en/stable/python/python_api.html>)
  * [LightGBM sklearn API tutorial](<https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>)
  

## Author pages<p>
The project has six contributors. All the page links will be refined in the future<p>
#### @ACSEKevin <https://github.com/ACSEkevin><br>

# DengAI_proposal
Project in development. The task consists of estimating the number of denge infections in two different cities using the data available at [DrivenDATA](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/).

The python code here can be useful for other projects related with temporal sequence analysis, although no specific techniques such as ARIMA/ARMA nor (Facebook) Prophet are being used.

## Versioning
v1.21:
* Forecasting was divided into modelling seasonality + anomalies. Such anomalies are modelled with an anomaly detection algorithm, which in the end contributes with seasonality to produce the output signal.

v1.20:
* Inspired in linear models.

v1.19:
* Modified the forecasting procedure. Using BayesianRidge.

v1.18:
* Started working in the prediction of the output unlabelled subset.

v1.17:
* Removed some of the warnings that appeared on the code, mainly with the comment: 'SettingWithCopyWarning'.

v1.16:
* Included some serius forecasting based on memory and using previous forecasted values.

v1.15:
* Added models that make use both features and previous forecasts as their inputs.

v1.14:
* Included new time series analysis.

v1.13:
* Cleaning the notebook.

v1.12:
* Code repaired so every model is able to work using scaling.

v1.11:
* I need to include the scaler for models such as the SVR, MLP and Dense. I should also scale the target variable.
* Removed logarithmic labelling because is not being used.
* Totally implemented the DNN from ML_utils and adapted to work in Main.py.

v1.10:
* Keeping only 5 features.
  * C:sj. M:Dense. Test:17.64572956206951
  * C:iq. M:Dense. Test:5.6923076923076925
  * DRIVENDATA: 27.8894

v1.9:
* Keeping only 5 features.
* In order to increase stability, we use 10 splits in TimeSeriesSplit.
  * C:sj. M:RandomForest. Test:18.98517824723589
  * C:iq. M:RandomForest. Test:4.144047298319872

v1.8:
* Retrain using 100% of the dataset.

v1.7:
* Retrain using 100% of the dataset.
* Keeping only 5 features.
* Forced to use SVR.
  * C:sj. M:SVR. Test:18.063648014200677
  * C:iq. M:SVR. Test:3.378613199666314
  * DRIVENDATA: 26.7091
  
v1.6:
* Explicit control of polynomial features.
* We force the usage of KNN.
  * C:sj. M:KNN. Test:21.701063829787238
  * C:iq. M:KNN. Test:4.818681318681319
  
v1.5c:
* Working on the notebook to make it look more appealing.

v1.5b:
* Decided to remove Adaboost from the pool. It seems to overfit strongly to data.
* Included MLP and GradientBoosting in the regressor pool.

v1.5a:
* Bigger learning rate in AdaBoost.
  * C:sj. M:AdaBoost. Test:17.627599802242994
  * C:iq. M:KNN. Test:4.6739010989011

v1.5:
* Included new features computed as polynomia from the previously selected ones.
* We need to select features for each of the cities. Apparently, feature correlation is different for the two scenarios under analysis.
  * C:sj. M:AdaBoost. Test:17.65192668194498
  * C:iq. M:KNN. Test: 4.6739010989011
  * DRIVENDATA: 30.4904

v1.4c:
* Same than v1.4 but only using KNN. More neighbors in the Regressor.
  * C:sj. M:KNN. Test:21.446143181514252
  * C:iq. M:KNN. Test:4.666208791208792

v1.4b:
* Same than v1.4 but only using KNN.
  * C:sj. M:KNN. Test:21.446143181514252
  * C:iq. M:KNN. Test:4.859615384615385
  * DRIVENDATA: 25.6587

v1.4a:
* Playing with some features generated with TSNE and Polynomia.

v1.4:
* Added more options to the ML Regressor params.
* Included analysis about the correlation of features and total_cases. We select the top 7. 
  * C:sj. M:AdaBoost. Test:18.021608985610648
  * C:iq. M:KernelRidge. Test:4.676544254552
  * DRIVENDATA: 26.7212

v1.3:
* Since we observe that there is a strong inverse correlation between the year and total_cases, we introduce inverse_year which is computed as 1/year.
* Included analysis about the correlation of features and total_cases. We select the top 7. 
  * C:sj. M:RandomForest. Test:17.260034414332367
  * C:iq. M:KernelRidge. Test:4.676544254552
  * DRIVENDATA: 26.7764

v1.2: 
* Season+Manual feature selection according to: http://drivendata.co/blog/dengue-benchmark/
  * C:sj. M:KernelRidge. Test:20.6502157735127
  * C:iq. M:KernelRidge. Test:4.669998207138845
  * DRIVENDATA: 25.8750

v1.1:
* KNN+Season+Manual feature selection according to: http://drivendata.co/blog/dengue-benchmark/
  * City1: KNN. Train: 0.0. Test: 22.011492034965695
  * City2: KNN. Train: 6.498504273504274. Test error: 4.862499999999999
  * DRIVENDATA: 25.3966

v1.0:
* Adaboost+Season+Regular features
  * DRIVENDATA: 30.1707

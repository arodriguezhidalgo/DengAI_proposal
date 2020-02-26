# DengAI_proposal
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

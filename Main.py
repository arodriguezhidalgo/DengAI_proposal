import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def read_data(data_file):
    data_structure = pd.read_csv(os.path.join('data',data_file))
    return data_structure

features = read_data('dengue_features_train.csv');
labels = read_data('dengue_labels_train.csv');
print('Training data readed!')

# We mix features and labels in a single dataset for commodity
features['total_cases']=labels['total_cases']

city_models = {}
for id_city in features.city.unique():
    city_data = features[features['city'] == id_city]
     
    # Data imputation
    missing = {}
    for col in city_data:
        missing_no = city_data[col].isna().sum()
        if missing_no != 0:
            missing[col] = missing_no;
    print('Missing data:')
    print(missing)
    
    ''' 
    We interpolate the missing values considering each city separatedly. We 
    take advantage of the fact that this is a temporal sequence, so interpolation
    seems reasonable.
    '''
    for col in missing.keys():
        city_data[col].interpolate(inplace=True)
        
    # We create some new features that might be interesting.
    city_data['USER_month'] = city_data.apply(lambda x: int(x['week_start_date'][5:7]), axis=1);
    city_data['USER_day'] = city_data.apply(lambda x: int(x['week_start_date'][8:10]), axis=1);
    city_data.pop('week_start_date');
    city_data.pop('city');
    
    # We define the four seasons from the North hemishpere.    
    from DengAI_utils import season
    city_data['USER_season'] = city_data['USER_month'].apply(season);
    
    # We generate logarithmic labels, which might be useful to model seasonality
    city_data['total_cases_LOG'] = np.log(city_data['total_cases'])
    city_data['total_cases_LOG'][city_data['total_cases_LOG'] <0 ] =0

    from DengAI_utils import compute_correlation
    compute_correlation(city_data);
    
    '''
    Although we are going to perform TimeSeriesSplit to perform our analysis, we
    keep some of the data out of the validation pool for testing purposes.
    
    Notice that we will use Logarithmic labels to perform training.
    '''
    
    
    x_train = city_data[[col for col in city_data.columns if col not in ['total_cases','total_cases_LOG','diff','pos_neg']]]
    y_train = city_data['total_cases_LOG']
    i_test = int(np.round(len(x_train.index))*.9); # 10 per cent used for test in any dataset
    
    x_test = x_train.iloc[i_test:]
    y_test = y_train[i_test:].values
    x_train = x_train.iloc[:i_test]
    y_train = y_train[:i_test].values
    
    from ML_utils.Regressors import Regressors
    n_splits = 5 # TimeSeriesSplits number of splits
    
    # We train a selection of models
    reg_list = ['RandomForest','KNN','AdaBoost','BayesianRidge','KernelRidge','LinearRegression'];#['RandomForest','KNN','GradientBoosting','AdaBoost','BayesianRidge','KernelRidge','LinearRegression'];
    model = {};
    model_scores = {};
    for model_name in reg_list:
        # Random Forest
        model[model_name] = Regressors()
        model[model_name].get_regressor(model_name)
        model[model_name].get_TimeSeries_CV(score='neg_mean_absolute_error',n_splits = n_splits)
        model[model_name].fit_model(x_train, y_train)
    
        y_pred_train = model[model_name].return_prediction(x_train)
        y_pred_test  = model[model_name].return_prediction(x_test)
    
        from sklearn.metrics import mean_absolute_error
        model_scores[model_name] = model[model_name].plot_results(np.exp(y_test), np.exp(y_pred_test), mean_absolute_error)
        plt.title(model_name);
    
    '''
    Using the previous models we train a meta-Regressor, which we hope it produces
    better results as a combination of the previous systems.
    '''
    y_pred_train = np.zeros_like(y_train)
    y_pred_test = np.zeros_like(y_test)
    for model_name in reg_list:
        y_pred_train += model[model_name].return_prediction(x_train)
        y_pred_test  += model[model_name].return_prediction(x_test)
    
    y_pred_train /= len(reg_list);
    y_pred_test /= len(reg_list);
    print('Train error: {}'.format(mean_absolute_error(np.exp(y_train), np.exp(y_pred_train))))
    print('Test error: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred_test))))
    
    plt.subplot(211)
    plt.plot(np.exp(y_pred_train))
    plt.plot(np.exp(y_train))
    plt.subplot(212)
    plt.plot(np.exp(y_pred_test))
    plt.plot(np.exp(y_test))

    
    city_models[id_city] = {};
    #city_models[id_city]['meta'] = meta;
    city_models[id_city]['meta_sub'] = model;
    city_models[id_city]['meta_sub_scores'] = model_scores;
    print('Finisthed with city: {}'.format(id_city))
print('Training finished!')

x_test_real = pd.read_csv('data/dengue_features_test.csv')
missing = {}
for col in x_test_real:
    missing_no = x_test_real[col].isna().sum()
    if missing_no != 0:
        missing[col] = missing_no;
        
for col in missing.keys():
    x_test_real[col].interpolate(inplace=True)

x_test_real['USER_month'] = x_test_real.apply(lambda x: int(x['week_start_date'][5:7]), axis=1);
x_test_real['USER_day'] = x_test_real.apply(lambda x: int(x['week_start_date'][8:10]), axis=1);
x_test_real.pop('week_start_date');
x_test_real['USER_season'] = x_test_real['USER_month'].apply(season);


#city_data = x_test_real[x_test_real['city'] == id_city]
output_df = pd.DataFrame(columns = ['city', 'year', 'weekofyear', 'total_cases'])
outcome = []
for i_row in x_test_real.index:
    aux = x_test_real.iloc[i_row]
    x_aux = aux[[col for col in aux.index if col not in ['city','total_cases','total_cases_LOG','diff','pos_neg']]]
    x_aux = np.expand_dims(x_aux, axis=0)
    
    #if aux['city'] == 'sj':
    model_aux = city_models[aux['city']]['meta_sub']['AdaBoost']
    y_aux = int(np.ceil(np.exp(model_aux.return_prediction(x_aux)[0])))
    output_df = output_df.append({   'city':         aux['city'],
                          'year':        aux['year'],
                          'weekofyear':  aux['weekofyear'],
                          'total_cases': y_aux
                      }, ignore_index = True)
    outcome.append(y_aux)
#    x_aux = 
output_df.to_csv('output.csv', index = False)
plt.plot(outcome)
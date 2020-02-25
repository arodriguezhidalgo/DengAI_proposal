import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def read_data(data_file):
    data_structure = pd.read_csv(os.path.join('data',data_file))
    return data_structure


logarithmic_labels = False;
feature_selection = True;
feature_vector = ['weekofyear',
                  'month',
                  'inverse_year',
                  'reanalysis_specific_humidity_g_per_kg',
                  'reanalysis_dew_point_temp_k',
                  'station_avg_temp_c',
                  'reanalysis_max_air_temp_k',
                  'total_cases']
verbose = False;

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
    city_data['month'] = city_data.apply(lambda x: int(x['week_start_date'][5:7]), axis=1);
    city_data['day'] = city_data.apply(lambda x: int(x['week_start_date'][8:10]), axis=1);
    city_data['inverse_year'] = city_data['year'].apply(lambda x: 1/x);
    city_data.pop('week_start_date');
    city_data.pop('city');
    
    # We define the four seasons from the North hemishpere.    
    from DengAI_utils import season
    city_data['season'] = city_data['month'].apply(season);
    
    if feature_selection == True:        
        city_data = city_data[feature_vector]
    
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
    if logarithmic_labels == True:
        y_train = city_data['total_cases_LOG']
    else: 
        y_train = city_data['total_cases']
        
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
        if logarithmic_labels == True:
            model_scores[model_name] = model[model_name].plot_results(np.exp(y_test), np.exp(y_pred_test), mean_absolute_error, verbose);
        else:
            model_scores[model_name] = model[model_name].plot_results(y_test, y_pred_test, mean_absolute_error, verbose)
            
        if verbose == True: 
            plt.title(model_name);
    
    '''
    Using the previous models we train a meta-Regressor, which we hope it produces
    better results as a combination of the previous systems.
    '''
    y_pred_train = np.zeros_like(y_train, dtype=float)
    y_pred_test = np.zeros_like(y_test, dtype=float)
    for model_name in reg_list:
        y_pred_train += model[model_name].return_prediction(x_train)
        y_pred_test  += model[model_name].return_prediction(x_test)
    
    y_pred_train /= len(reg_list);
    y_pred_test /= len(reg_list);
    if logarithmic_labels == True:
        print('Train error: {}'.format(mean_absolute_error(np.exp(y_train), np.exp(y_pred_train))))
        print('Test error: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred_test))))
        if verbose == True:
            plt.subplot(211)
            plt.plot(np.exp(y_pred_train))
            plt.plot(np.exp(y_train))
            plt.subplot(212)
            plt.plot(np.exp(y_pred_test))
            plt.plot(np.exp(y_test))
    else:
        print('Train error: {}'.format(mean_absolute_error(y_train, y_pred_train)))
        print('Test error: {}'.format(mean_absolute_error(y_test, y_pred_test)))
        if verbose == True:
            plt.subplot(211)
            plt.plot(y_pred_train)
            plt.plot(y_train)
            plt.subplot(212)
            plt.plot(y_pred_test)
            plt.plot(y_test)


    
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
    
x_test_real['month'] = x_test_real.apply(lambda x: int(x['week_start_date'][5:7]), axis=1);
x_test_real['day'] = x_test_real.apply(lambda x: int(x['week_start_date'][8:10]), axis=1);
x_test_real['inverse_year'] = x_test_real['year'].apply(lambda x: 1/x);
x_test_real['season'] = x_test_real['month'].apply(season);

if feature_selection == True:        
    extra_column = x_test_real[['year','weekofyear','city']];
    x_test_real = x_test_real[[i for i in feature_vector if i not in ['total_cases']]]





#city_data = x_test_real[x_test_real['city'] == id_city]
output_df = pd.DataFrame(columns = ['city', 'year', 'weekofyear', 'total_cases'])
outcome = []
for i_row in x_test_real.index:
    aux = x_test_real.iloc[i_row]
    x_aux = aux[[col for col in aux.index if col not in ['city','total_cases','total_cases_LOG','diff','pos_neg']]]
    x_aux = np.expand_dims(x_aux, axis=0)
    
    # We always use the model that produces minimum test score for each city.    
    model_name = min(city_models[extra_column.iloc[i_row]['city']]['meta_sub_scores'], key=city_models[extra_column.iloc[i_row]['city']]['meta_sub_scores'].get);
    model_aux = city_models[extra_column.iloc[i_row]['city']]['meta_sub'][model_name]
    print('**C:{}. M:{}. Test:{}'.format(extra_column.iloc[i_row]['city'],model_name,city_models[extra_column.iloc[i_row]['city']]['meta_sub_scores'][model_name]))
    if logarithmic_labels == True:
        y_aux = int(np.ceil(np.exp(model_aux.return_prediction(x_aux)[0])))
    else:
        y_aux = int(np.ceil(model_aux.return_prediction(x_aux)[0]))
        

    output_df = output_df.append({   
                          'city':         extra_column.iloc[i_row]['city'],
                          'year':        extra_column.iloc[i_row]['year'],#aux['year'],
                          'weekofyear':  extra_column.iloc[i_row]['weekofyear'],
                          'total_cases': y_aux
                      }, ignore_index = True)
    outcome.append(y_aux)
#    x_aux = 
output_df.to_csv('output.csv', index = False)
plt.figure()
plt.plot(outcome)
print('Code finished.')
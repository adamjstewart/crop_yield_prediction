import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

global yield_type_dict
yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}

# Variable combinations
var_base = {'T_linear_6_8': " + tave6 + tave7 + tave8",
            
            'P_linear_6_9': " + precip6 + precip7 + precip8 + precip9",
            
            'VPD_linear_6_8': " + vpdave6 + vpdave7 + vpdave8",
            
            'T_poly_6_8': (" + tave6 + tave7 + tave8"  
                           "+ np.power(tave6, 2) + np.power(tave7, 2) + np.power(tave8, 2)"),
            
            'P_poly_6_9': (" + precip6 + precip7 + precip8 + precip9" +
                           "+ np.power(precip6, 2) + np.power(precip7, 2) + np.power(precip8, 2) + np.power(precip9, 2)"),
            
            'VPD_poly_6_8': ("+ vpdave6 + vpdave7 + vpdave8" + 
                             " + np.power(vpdave6, 2) + np.power(vpdave7, 2) + np.power(vpdave8, 2)"),
            
            'LSTMAX_linear_6_8': "+ lstmax6 + lstmax7 + lstmax8",
            
            'LSTMAX_poly_6_8': ("+ lstmax6 + lstmax7 + lstmax8" + 
                                "+ np.power(lstmax6, 2) + np.power(lstmax7, 2) + np.power(lstmax8, 2)"),
            
            'T_spline_6_8': ("+ bs(tave6, df=3, knots = (20,23), degree=1,lower_bound=7,upper_bound=35)"
                            + "+ bs(tave7, df=3, knots = (22,26), degree=1,lower_bound=10,upper_bound=40)" 
                            + "+ bs(tave8, df=3, knots = (20,24), degree=1,lower_bound=11,upper_bound=40)"), 
                                
            'VPD_spline_6_8': ("+ bs(vpdave6, df=5, knots = (8,10,13,15), degree=1,lower_bound=4,upper_bound=30)" 
                              + "+ bs(vpdave7, df=3, knots = (8,11), degree=1,lower_bound=4,upper_bound=35)" 
                              + " + bs(vpdave8, df=3, knots = (8,15), degree=1,lower_bound=3,upper_bound=30)"),
                                
            'P_spline_6_9': (" + bs(precip6, df=3, knots = (75,200), degree=1,lower_bound=0,upper_bound=500)"
                            + " + bs(precip7, df=3, knots = (75,200), degree=1,lower_bound=0,upper_bound=600)"
                            + " + bs(precip8, df=2, knots = (90,), degree=1,lower_bound=0,upper_bound=500)"
                            + " + bs(precip9, df=3, knots = (100,200), degree=1, lower_bound=0, upper_bound=500)"),
            
            'LSTMAX_spline_6_8':("+ bs(lstmax6, df=3, knots = (28,34), degree=1,lower_bound=20,upper_bound=50)" 
                                + "+ bs(lstmax7, df=4, knots = (26,31,35), degree=1,lower_bound=20,upper_bound=53)" 
                                + "+ bs(lstmax8, df=3, knots = (25,28), degree=1, lower_bound=18, upper_bound=48)"),
            
            'EVI_linear_5_9': "+ evi5 + evi6 + evi7 + evi8 + evi9",
            
            'EVI_poly_5_9': ("+ evi5 + evi6 + evi7 + evi8 + evi9"
                             + " + np.power(evi5,2) + np.power(evi6, 2) + np.power(evi7, 2) + np.power(evi8, 2)"
                             + " + np.power(evi9, 2)"),
            
            'EVI_spline_5_9': (" + bs(evi5, df=3, knots= (0.25,0.35), degree=1,upper_bound=0.8)"
                             + " + bs(evi6, df=3, knots= (0.43,0.5), degree=1,upper_bound=0.8)"
                             + " + bs(evi7, df=3, knots= (0.5,0.65), degree=1,upper_bound=0.8)"
                             + " + bs(evi8, df=3, knots= (0.55,0.65), degree=1,upper_bound=0.8)"
                             + " + bs(evi9,df=3, knots= (0.26,0.36), degree=1,upper_bound=0.8)"),

            'Tgs_linear': (" + tave56789 + precip56789"), 
            
            'Tgs_poly': (" + tave56789 + np.power(tave56789, 2)"
                         + " + precip56789 + np.power(precip56789, 2)")         
           }

"""
Constract different models using the variable bases, return model_txt
model_txt = define_model_structure_test('vpd_spline_evi', yield_type='rainfed') 
"""
def define_model_structure_test(name, yield_type='all'):
#     yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
    
    if name == 'tave_linear':
        model_vars = var_base['T_linear_6_8'] + var_base['P_linear_6_9']

    if name == 'Tgs_linear':
        model_vars = var_base['Tgs_linear']
    
    if name == 'tave_spline':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9']
        
    if name == 'tave_poly':
        model_vars = var_base['T_poly_6_8'] + var_base['P_poly_6_9']

    if name == 'Tgs_poly':
        model_vars = var_base['Tgs_poly']
        
    if name == 'vpd_linear':
        model_vars = var_base['VPD_linear_6_8'] + var_base['P_linear_6_9']
        
    if name == 'vpd_poly':
        model_vars = var_base['VPD_poly_6_8'] + var_base['P_poly_6_9']    
        
    if name == 'vpd_spline':
        model_vars = var_base['VPD_spline_6_8'] + var_base['P_spline_6_9']       

    if name == 'lstmax_linear_only':
        model_vars = var_base['LSTMAX_linear_6_8']
        
    if name == 'lstmax_poly_only':
        model_vars = var_base['LSTMAX_poly_6_8']
        
    if name == 'lstmax_spline_only':
        model_vars = var_base['LSTMAX_spline_6_8']   
        
    if name == 'evi_linear_only':
        model_vars = var_base['EVI_linear_5_9']  
        
    if name == 'evi_poly_only':
        model_vars = var_base['EVI_poly_5_9']     
        
    if name == 'evi_spline_only':
        model_vars = var_base['EVI_spline_5_9']    
        
    if name == 'tave_spline_evi':
        model_vars =var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_linear_5_9'] 
        
    if name == 'tave_spline_evi_poly':
        model_vars =var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5_9'] 
        
    if name == 'tave_poly_evi':
        model_vars =var_base['T_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_linear_5_9']      
        
    if name == 'vpd_spline_evi':
        model_vars =var_base['VPD_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_linear_5_9']  
        
    if name == 'vpd_poly_evi':
        model_vars =var_base['VPD_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_linear_5_9']   
        
    if name == 'vpd_poly_evi_poly':
        model_vars =var_base['VPD_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_poly_5_9']    
        
    if name == 'vpd_spline_evi_poly':
        model_vars =var_base['VPD_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5_9']         
        
    if name == 'lstmax_spline_evi':
        model_vars = var_base['LSTMAX_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_linear_5_9'] 
        
    if name == 'lstmax_poly_evi_poly':
        model_vars = var_base['LSTMAX_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_poly_5_9']

    if name == 'lstmax_poly_evi_poly_only':
        model_vars = var_base['LSTMAX_poly_6_8'] + var_base['EVI_poly_5_9']
        
    if name == 'lstmax_spline_evi_poly':
        model_vars = var_base['LSTMAX_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5_9'] 
        
    if name == 'lstmax_spline_evi_poly_only':
        model_vars = var_base['LSTMAX_spline_6_8'] + var_base['EVI_poly_5_9']         
        
    return ("Q('%s_ana') ~ "%yield_type_dict[yield_type] + model_vars + "+ C(FIPS)")

"""
Train a model and do the prediction
m, df_predicted = init_model(df_train, df_test, model_txt,yield_type='rainfed')
"""
def init_model(df_train, df_test, model_txt,yield_type='rainfed',weight=False):
    if weight:
        results = smf.wls(model_txt, data=df_train, missing='drop',weights=df_train['corn_percent']).fit()
    else:
        results = smf.ols(model_txt, data=df_train,missing='drop').fit()
    return results, df_test.copy().join(results.predict(df_test).to_frame('Predicted_'+yield_type_dict[yield_type]+'_ana'))

"""
Estimate the global yield trend
"""
def yield_trend(df, yield_type='rainfed'):
#     yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
    # Estimate regional yield trend and detrend
    trend_model_txt = "Q('%s')"%yield_type_dict[yield_type] + "~ year"
    trend_results = smf.ols(trend_model_txt, data=df).fit()
    return trend_results

"""
Build the model and predict a given year
"""
def get_prediction_for_year(df, model_type, y, yield_type, prediction_type='forward',
                            state_subset=False,fix=False):

    model_txt = define_model_structure_test(model_type, yield_type=yield_type)
    # Fix svd issue by filering counties 
    if fix:
        corn_percent_min = 0.001
        print('model %s for year %d with fix'%(model_type,y))
    else:
        corn_percent_min = 0
        print('model %s for year %d'%(model_type,y))

    if prediction_type == 'forward':
        train_data = df[(df['year']<y)&(df['corn_percent']>corn_percent_min)]
#        train_data = df[(df['year']<y)]

    if prediction_type == 'leave_one_year_out':
        train_data = df[(df['year']!=y)&(df['corn_percent']>corn_percent_min)]
#        train_data = df[(df['year']!=y)]

    test_data = df[(df['year']==y)&(df['corn_percent']>corn_percent_min)]
#    test_data = df[(df['year']==y)]


    if state_subset: 
        train_data = train_data[(train_data['State'].isin(state_subset))]
        test_data = test_data[(test_data['State'].isin(state_subset))]

     
    trend_results = yield_trend(df, yield_type=yield_type)

    # only predict county in train data
    con_fips = test_data['FIPS'].isin(train_data['FIPS'].unique())

    # If predict irrigated yield but there is no valid data in the training, 
    # E.g., Illinois after the state subset, set the predicted values to be nan
    if (train_data['yield_irr'].isnull().all())&(yield_type=='irrigated'):
        df_predict = test_data[con_fips].copy()
        df_predict['Predicted_'+yield_type_dict[yield_type]] = np.nan
    else:
        m, df_predict = init_model(train_data, test_data[con_fips], model_txt, yield_type=yield_type)
        df_predict['Predicted_'+yield_type_dict[yield_type]] = (df_predict['Predicted_'+yield_type_dict[yield_type]+'_ana']
                                                                + trend_results.predict(df_predict['year']))
    return df_predict


"""
Build model, make prediction and test the performance
Example: result_linear_climate = get_prediction(data, 'linear_climate',yield_type='irrigated')
"""
def get_prediction_year_range(df, model_type, prediction_type='forward', yield_type='rainfed', 
                   state_subset=False, year_range=[2003,2016],fix=False):
    
    test_start_year = year_range[0]
   # print('start test year is %d'%test_start_year)    
    
    if ('evi' in model_type)&(prediction_type=='forward'):
        test_start_year = 2005
    
    if ('lstmax' in model_type) & (prediction_type=='forward'):
        test_start_year = 2005
   # elif ('lstmax' in model_type) & (prediction_type=='leave_one_year_out'):
   #     test_start_year = 2003
   # print('start test year is %d'%test_start_year)
    
    if yield_type != 'all':
        frame = [get_prediction_for_year(df, model_type, i, yield_type,prediction_type=prediction_type,
                    state_subset=state_subset,fix=fix) for i in range(test_start_year, year_range[1]+1)]
        return pd.concat(frame)
    else:
        frame1 = [get_prediction_for_year(df, model_type, i, 'rainfed',prediction_type=prediction_type,
                    state_subset=state_subset,fix=fix) for i in range(test_start_year, year_range[1]+1)]
        
        frame2 = [get_prediction_for_year(df, model_type, i, 'irrigated',prediction_type=prediction_type,
            state_subset=state_subset,fix=fix) for i in range(test_start_year, year_range[1]+1)]
        
        df_out = pd.concat(frame1).merge(pd.concat(frame2).loc[:,('year','FIPS','Predicted_yield_irr_ana','Predicted_yield_irr')])
        
        df_out['Predicted_yield'] = (df_out.fillna(0)['Predicted_yield_rainfed'] * df_out.fillna(0)['area_rainfed'] 
            + df_out.fillna(0)['Predicted_yield_irr'] * df_out.fillna(0)['area_irr'])/(df_out.fillna(0)['area_rainfed']
                                                                                       + df_out.fillna(0)['area_irr'])
        df_out.loc[df_out['Predicted_yield']==0, 'Predicted_yield'] = np.nan # otherwise the value is zero
        return df_out
    


def load_yield_data():
    data = pd.read_csv('../data/Corn_model_data.csv',dtype={'FIPS':str})
    
    data['corn_percent'] = data['area']/data['land_area']
    
    # Add logical filter to the yield Data
    area_con = data['area'].notnull()
    data = data[area_con]
    
    # Add Rainfed yield
    # rainfed_con: counties without irrigation, the yield is rainfed
    rainfed_con = ~data['FIPS'].isin(data.loc[data['yield_irr'].notnull(),'FIPS'].unique())
    data['yield_rainfed'] = data['yield_noirr']
    data['area_rainfed'] = data['area_noirr']
    
    
    # For counties with irrigation, only the rainfed yield is added to irrigated yield
    data.loc[rainfed_con, 'yield_rainfed'] = data.loc[rainfed_con, 'yield']
    data.loc[rainfed_con, 'area_rainfed'] = data.loc[rainfed_con, 'area']

    # add growing season
    data['tave56789']= data.loc[:,'tave5':'tave9'].mean(axis=1)
    data['vpdave56789']= data.loc[:,'vpdave5':'vpdave8'].mean(axis=1)
    data['precip56789']= data.loc[:,'precip5':'precip9'].sum(axis=1)
    
    
    # Add z-score
    county_std = data.groupby('FIPS').std()['precip56789'].to_frame('precip_gs_std').reset_index()
    county_mean = data.groupby('FIPS').mean()['precip56789'].to_frame('precip_gs_mean').reset_index()
    
    data = data.merge(county_mean, on='FIPS').merge(county_std, on='FIPS')
    
    data['precip_gs_z'] = (data['precip56789'] - data['precip_gs_mean'])/data['precip_gs_std']

    # The 12 core states 
    data_12 = data[data['State'].isin(data.loc[data['evi6'].notnull(),'State'].unique())]

    # Detrend yield
    global trend_rainfed, trend_irrigated, trend_all
    trend_rainfed = yield_trend(data_12, yield_type='rainfed')
    trend_irrigated = yield_trend(data_12, yield_type='irrigated')
    trend_all = yield_trend(data_12, yield_type='all')
    
    data_12.loc[:,'yield_ana'] = (data_12['yield'] - trend_all.predict(data_12[['year','yield']]))
    data_12.loc[:,'yield_rainfed_ana'] = (data_12['yield_rainfed'] - trend_rainfed.predict(data_12[['year','yield_rainfed']]))      
    data_12.loc[:,'yield_irr_ana'] = (data_12['yield_irr'] - trend_irrigated.predict(data_12[['year','yield_irr']])) 
    
    return data_12

def save_prediction(df,yield_type='rainfed',prediction_type='forward'):
    models = ['Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly',#4
              ]
    svd_issue=['lstmax_spline_evi_poly_only','vpd_spline','vpd_spline_evi','vpd_spline_evi_poly'] # all, leave_one_year_out
   # svd_issue = ['vpd_spline_evi_poly','vpd_spline']
   # svd_issue = ['lstmax_spline_evi_poly','tave_spline_evi','tave_spline_evi_poly'] # for all, forward
    # vpd_spline_evi_poly, and vpd_spline have the svd problem for leave_one_year_out
    for m in models[20::]:
        print('Saving prediction for %s, %s, %s'%(m,yield_type,prediction_type))
        if m in svd_issue:
            df_g = get_prediction_year_range(df, m, yield_type=yield_type, prediction_type=prediction_type,fix=True)
        else:
            df_g = get_prediction_year_range(df, m, yield_type=yield_type, prediction_type=prediction_type)
        df_g.to_csv('../data/result/prediction_%s_%s_%s.csv'%(m,yield_type,prediction_type)) 
#        df_g.to_csv('../data/result/prediction_%s_%s_%s_weighted_regression.csv'%(m,yield_type,prediction_type)) 
#        df_g.to_csv('../data/result/prediction_%s_%s_%s_cornpercent.csv'%(m,yield_type,prediction_type)) 

# Save prediction trained from each state
def save_prediction_state(df,yield_type='rainfed',prediction_type='forward'):
    models = ['vpd_poly_evi_poly','tave_poly_evi','vpd_poly','tave_poly']
    states = df['State'].unique()
    for m in models[2:4]:
        frame = [get_prediction_year_range(df, m,yield_type=yield_type, prediction_type=prediction_type,
                 state_subset=[s]) for s in states]
        pd.concat(frame).to_csv('../data/result/prediction_%s_%s_%s_state.csv'%(m,yield_type,prediction_type))
        print('Saving prediction by state for %s, %s, %s'%(m,yield_type,prediction_type))
#    return pd.concat(frame)

# Calculate global prediction performance 
def prediction_result_global(test,yield_type='rainfed'):
    result = pd.DataFrame(np.full([test['year'].unique().shape[0],3], np.nan), index=test['year'].unique(),
                          columns=['R2','rmse','R2_classic'])

    for y in range(test['year'].min(), test['year'].max()+1):
        con = test['year']==y
        r2_temp = test.loc[con,[yield_type_dict[yield_type],
                                'Predicted_'+yield_type_dict[yield_type]]].corr() \
            ['Predicted_'+yield_type_dict[yield_type]][0]**2
            
        rmse_temp = (((test.loc[con, 'Predicted_'+yield_type_dict[yield_type]] - 
                          test.loc[con, yield_type_dict[yield_type]])**2).sum() \
                                      /test.loc[con,yield_type_dict[yield_type]].shape[0])**0.5
                     
#                                       /test.loc[con,'Predicted_'+yield_type_dict[yield_type]].shape[0])**0.5

        sst = ((test.loc[con, yield_type_dict[yield_type]] 
                - test.loc[con, yield_type_dict[yield_type]].mean())**2).sum()
        sse = ((test.loc[con, yield_type_dict[yield_type]] - test.loc[con, 'Predicted_'+yield_type_dict[yield_type]])**2).sum()

        result.loc[y] = [r2_temp, rmse_temp, 1-sse/sst]
    return result  

# Get the prediction results at each state
def prediction_result_by_state(test,yield_type='rainfed'):

    # Create a multi-index dataframe to save state results
    iterables = [test['State'].unique(), test['year'].unique()]
    idx = pd.MultiIndex.from_product(iterables, names=['State', 'year'])
    result = pd.DataFrame(np.full([idx.shape[0],3], np.nan), index=idx, columns=['R2','rmse','R2_classic'])

    for y in range(test['year'].min(), test['year'].max()+1):
        for st in test['State'].unique():
            con = (test['year']==y)&(test['State']==st)
            
            
            r2_temp = test.loc[con,[yield_type_dict[yield_type],
                                'Predicted_'+yield_type_dict[yield_type]]].corr() \
            ['Predicted_'+yield_type_dict[yield_type]][0]**2
            
            rmse_temp = (((test.loc[con, 'Predicted_'+yield_type_dict[yield_type]] - 
                          test.loc[con, yield_type_dict[yield_type]])**2).sum() \
                                      /test.loc[con,yield_type_dict[yield_type]].shape[0])**0.5
                     
#                                       /test.loc[con,'Predicted_'+yield_type_dict[yield_type]].shape[0])**0.5

            sst = ((test.loc[con, yield_type_dict[yield_type]] 
                    - test.loc[con, yield_type_dict[yield_type]].mean())**2).sum()
            sse = ((test.loc[con, yield_type_dict[yield_type]] - test.loc[con, 'Predicted_'+yield_type_dict[yield_type]])**2).sum()

            result.loc[st,y] = [r2_temp, rmse_temp, 1-sse/sst]
    return result        

if __name__=='__main__':
    df_12 = load_yield_data()
    save_prediction(df_12, yield_type='all',prediction_type='leave_one_year_out')
   # State level model prediction
#    save_prediction_state(df_12,yield_type='rainfed',prediction_type='forward')

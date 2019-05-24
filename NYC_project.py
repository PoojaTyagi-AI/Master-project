#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:42:20 2019

@author: poojatyagi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from haversine import haversine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_notebook, show



train_data = pd.read_csv("/Users/poojatyagi/Desktop/AI PROJECT/DATA SOURCES/train.csv")
test_data = pd.read_csv("/Users/poojatyagi/Desktop/AI PROJECT/DATA SOURCES/test.csv")
train_data.head()
train_data.shape

#numer of unique Id'S in dataset
print("There are %d unique id's in Training dataset, which is equal to the number of records"%(train_data.id.nunique()))

#checking for NaN values in a dataset
train_data.isnull().sum()

#checking teh datatype of each input feature
train_data.dtypes

#converting the datatype of "pickup_datetime" and "dropoff_datetime" into datetime format
train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])
train_data['dropoff_datetime'] = pd.to_datetime(train_data['dropoff_datetime'])

#Adding more features i.e. "weekday", "month", "weekday_num" and "pickup_hour"
train_data['weekday'] = train_data.pickup_datetime.dt.weekday_name
train_data['month'] = train_data.pickup_datetime.dt.month
train_data['weekday_num'] = train_data.pickup_datetime.dt.weekday
train_data['pickup_hour'] = train_data.pickup_datetime.dt.hour

train_data.head()

#Defining dist_calc() function for Calculating distance between  pickup and dropoff coordinates using Haversine formula
def dist_calc(df):
    pickup = (df['pickup_latitude'], df['pickup_longitude'])
    drop = (df['dropoff_latitude'], df['dropoff_longitude'])
    return haversine(pickup, drop)


#Calculate distance and assign new feature "haversin_-distance" to the dataframe.
train_data['haversin_distance'] = train_data.apply(lambda x: dist_calc(x), axis = 1)


#Calculate Speed in km/h for further insights
train_data['speed'] = (train_data.haversin_distance/(train_data.trip_duration/3600))

#Check the type of each input feature
train_data.dtypes.reset_index()


'''Treating categorical features Dummify all the categorical features like "store_and_fwd_flag, vendor_id, 
   month, weekday_num, pickup_hour, passenger_count" except the label i.e. "trip_duration" '''

dummy = pd.get_dummies(train_data.store_and_fwd_flag, prefix='flag')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
train_data = pd.concat([train_data,dummy], axis = 1)

dummy = pd.get_dummies(train_data.vendor_id, prefix='vendor_id')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
train_data = pd.concat([train_data,dummy], axis = 1)

dummy = pd.get_dummies(train_data.month, prefix='month')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
train_data = pd.concat([train_data,dummy], axis = 1)

dummy = pd.get_dummies(train_data.weekday_num, prefix='weekday_num')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
train_data = pd.concat([train_data,dummy], axis = 1)

dummy = pd.get_dummies(train_data.pickup_hour, prefix='pickup_hour')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
train_data = pd.concat([train_data,dummy], axis = 1)

dummy = pd.get_dummies(train_data.passenger_count, prefix='passenger_count')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
train_data = pd.concat([train_data,dummy], axis = 1)

############################# UNIVARIATE ANALYSIS ######################################################################################

'''[1.] PASSENGER_COUNT (According to NY Limousine commsion 5 adults and 1 child is the max limit of taxi passenger)'''
train_data.passenger_count.value_counts()

#Detecting outliers (here passenge count = 7,8 ,9)
plt.figure(figsize = (20,5))
sns.boxplot(train_data.passenger_count)
plt.show() 
'''observations --1. Instances with 0 passengers exists
                  2. Trips with 7,8 and 9 passengers count is less
                  3. Maximum trips has passenger count 1 or 2 '''



#Treating outliers "passenger_count" feature
train_data.passenger_count.describe()
#coverting all the taxi trips with passenger count = 0 to passenger count = 1. since mean value is approx 1
train_data['passenger_count'] = train_data.passenger_count.map(lambda x: 1 if x == 0 else x)
#Considering trips with maximum passenger 6
train_data = train_data[train_data.passenger_count <= 6]

sns.countplot(train_data.passenger_count)
plt.show() #shows max trips with passenger 1 and max passenger count is 6


'''[2.] VENDOR '''
sns.countplot(train_data.vendor_id)
plt.show() #shows vendor 2 is more famous than vendor 1
 

'''[ 3.] Distance'''
print(train_data.haversin_distance.describe())

#Boxplot presentation of haversin_distance
plt.figure(figsize = (20,5))
sns.boxplot(train_data.haversin_distance)
plt.show()
'''OBSERVATION--1. There are trips with 0 km distance 
                2. Maximum trips are btw 0 -10 km
                3. There are few trips above 100 km
                4. Mean is 3.44
                5. Standard deviation is 4.30 so max trips have distance btw 0-10 km'''

print("Number of trips with zero distance are: ",train_data.haversin_distance[train_data.haversin_distance == 0 ].count())

#Treating outliers removing trips beyond 100 km
train_data = train_data[train_data.haversin_distance <= 100]
train_data.haversin_distance.groupby(pd.cut(train_data.haversin_distance, np.arange(0,100,10))).count().plot(kind='barh')
plt.show()



'''[4.] Trip duration'''
train_data.trip_duration.describe()

#Detecting outliers
plt.figure(figsize = (20,5))
sns.boxplot(train_data.trip_duration)
plt.show()

#OBSERVATIONS -----  1. There are few trips whose duration is beyond 86400 seconds i.e. 24 hours which is clear outliers
train_data = train_data[train_data.trip_duration <= 86400]

#visualizing the trip duration
train_data.trip_duration.groupby(pd.cut(train_data.trip_duration, np.arange(1,7200,600))).count().plot(kind='barh')
plt.xlabel('Trip Counts')
plt.ylabel('Trip Duration (seconds)')
plt.show()

'''OBSERVATIONS -- 1. Maximum trips have duration between 1-601 sec i.e. (0 -30 mins )'''



'''[5.] SPEED'''
'''Maximum speed limit in NYC is as follows:
   25 mph in urban area i.e. 40 km/h
   65 mph on controlled state highways i.e. approx 104 km/h'''

train_data.speed.describe()
plt.figure(figsize = (20,5))
sns.boxplot(train_data.speed)
plt.show()

'''Observation -- 1. Some trips are going beyond a aspeed of 220km which is not acceptable
                     So these are outliers and we need to remove them '''

train_data = train_data[train_data.speed <= 104]
plt.figure(figsize = (20,5))
sns.boxplot(train_data.speed)
plt.show()                     
                     
#Showing the speed in a range format
train_data.speed.groupby(pd.cut(train_data.speed, np.arange(0,104,10))).count().plot(kind = 'barh')
plt.xlabel('Trip count')
plt.ylabel('Speed (Km/H)')
plt.show()
'''Observation -- 1. Maximum trips have speed btw 10-20 km/hr'''

''' [6.] Store_and_fwd_flag (This flag indicates whether the trip record was held in vehicle
         memory before sending to the vendor because the vehicle did not have a connection
         to the server - Y=store and forward; N=not a store and forward trip.)'''

train_data.flag_Y.value_counts(normalize=True)
'''OBSERVATION --- It shows only 1% of the trip details were stored in the vehicle first before sending it to the server '''


'''[7.] Total trips per hour '''
sns.countplot(train_data.pickup_hour)
plt.show()

#observation -- There is a general trend of increasing taxi from 6AM and till 8PM 

''' [8.] Total trips per weak'''
plt.figure(figsize = (8,6))
sns.countplot(train_data.weekday_num)
plt.xlabel(' weekdays ')
plt.ylabel('Pickup counts')
plt.show()
'''Observation --- General trend of increasing taxi demand from moday till friday and
                   the decreasing on saturday and sunday.'''
#problem: On Monday there should be maximum trips spike as it is start of the week but trend shows it starts increasing from monday to friday
                   
#Showing patterns of taxi hourwise pickup pattern across the week
n = sns.FacetGrid(train_data, col='weekday_num')
n.map(plt.hist, 'pickup_hour')
plt.show()                   
''' OBSERVATIONS --- [1] Taxi pickups increased in the late night hours over the weekend possibly due to more outstation rides
                         or for the late night leisures nearby activities.
                     [2]Early morning pickups i.e before 5 AM have increased over the weekend in comparison to the office hours 
                         pickups i.e. after 7 AM which have decreased due to obvious reasons.
                     [3]Taxi pickups seems to be consistent across the week at 15 Hours i.e. at 3 PM. '''
                     
#problem: trend on sundays and mondays are same which is questionable.
         #On friday(weekday) during office hours taxi demand is reduced which is again questionable.
             
'''BIVARIATE ANALYSIS'''
'''[1.] TRIP DURATION PER HOUR'''
G1 = train_data.groupby('pickup_hour').trip_duration.mean()
sns.pointplot(G1.index, G1.values)
plt.ylabel('Trip Duration (sec)')
plt.xlabel('Pickup Hour')
plt.show()

'''[2.]Trip duration per weekday'''
G2 = train_data.groupby('weekday_num').trip_duration.mean()
sns.pointplot(G2.index, G2.values)
plt.ylabel('Trip Duration (sec)')
plt.xlabel('Weekday')
plt.show()    

'''[3.]Trip duration per month'''
G3 = train_data.groupby('month').trip_duration.mean()
sns.pointplot(G3.index, G3.values)
plt.ylabel('Trip Duration (sec)')
plt.xlabel('month')
plt.show()
''' 1.We can see an increasing trend in the average trip duration along with each subsequent month.
    2.The duration difference between each month is not much. It has increased gradually over a period of 6 months.
    3.It is lowest during february when winters starts declining.
    4.There might be some seasonal parameters like wind/rain which can be a factor of this gradual
          increase in trip duration over a period. Like May is generally the considered as the wettest month in NYC and 
          which is inline with our visualization. As it generally takes longer on the roads due to traffic jams during rainy season.
          So natually the trip duration would increase towards April May and June.'''
    

'''[4.]Trip duration per vendor'''
G4 = train_data.groupby('vendor_id').trip_duration.mean()
sns.barplot(G4.index, G4.values)
plt.ylabel('Trip Duration (sec)')
plt.xlabel('Vendor')
plt.show()
'''Average trip duration for vendor 2 is higher than vendor 1
 by approx 200 seconds i.e. atleast 3 minutes per trip.'''

'''[5.]Trip duration by flag'''
plt.figure(figsize = (6,5))
plot_dur = train_data.loc[(train_data.trip_duration < 10000)]
sns.boxplot(x = "flag_Y", y = "trip_duration", data = plot_dur)
plt.show()


'''[6.] Distance per hour'''
G6 = train_data.groupby('pickup_hour').haversin_distance.mean()
sns.pointplot(G6.index,G6.values)
plt.ylabel('Haversine Distance(Km)')
plt.xlabel('Pickup Hour')
plt.show()

'''[7.]trip count by vendor_id, flag_Y'''
train_data.groupby('flag_Y').vendor_id.value_counts().reset_index(name='count').pivot("flag_Y","vendor_id","count").plot(kind='bar')
plt.show()



'''[8.]distance v/s trip duration'''
plt.scatter(train_data.trip_duration, train_data.haversin_distance , s=1, alpha=0.5)
plt.ylabel('Distance')
plt.xlabel('Trip Duration')
plt.show()

#focus on the graph area where distance is < 60 km and duration is < 1500 seconds.
dur_dist = train_data.loc[(train_data.haversin_distance < 60) & (train_data.trip_duration < 1500), ['haversin_distance','trip_duration']]
plt.scatter(dur_dist.trip_duration, dur_dist.haversin_distance , s=1, alpha=0.5)
plt.ylabel('haversin_Distance')
plt.xlabel('Trip Duration')
plt.show()
'''observation : There should have been a linear relationship between 
the distance covered and trip duration on an average but we can see 
dense collection of the trips in the lower right corner which showcase 
many trips with the inconsistent readings.'''
 
'''SOLUTION ======> We should remove those trips which covered 0 km distance 
but clocked more than 1 minute to make our data more consistent for predictive model.
 Because if the trip was cancelled after booking, than that should not have taken more 
 than a minute time. This is our assumption.'''
 
train_data = train_data[~((train_data.haversin_distance == 0) & (train_data.trip_duration >= 60))]
train_data = train_data[~((train_data['haversin_distance'] <= 1) & (train_data['trip_duration'] >= 3600))]

'''[9.] Average speed per hour'''''
G9 = train_data.groupby('pickup_hour').speed.mean()
sns.pointplot(G9.index, G9.values)
plt.show()

'''[10.]Average speed per weekday'''
G10 = train_data.groupby('weekday_num').speed.mean()
sns.pointplot(G10.index, G10.values)
plt.show()

'''[11.]vendor_id v/s passenger count'''
G11 = train_data.groupby('vendor_id').passenger_count.mean()
sns.barplot(G11.index, G11.values)
plt.ylabel('Passenger count')
plt.show()

train_data.groupby('passenger_count').vendor_id.value_counts().reset_index(name='count').pivot("passenger_count","vendor_id","count").plot(kind='bar')
plt.show()
'''OBSERVATION ====> It seems that most of the big cars are served by the Vendor 2 
including minivans because other than passenger 1, vendor 2 has majority in serving 
more than 1 passenger count and that explains it greater share of the market.'''

train_data.dtypes
features_list_1 = list(zip(train_data.columns)) ##### Storing features list



''' Traffic visualization '''

train_fr = pd.read_csv('/Users/poojatyagi/Desktop/AI PROJECT/DATA SOURCES/osrmnyctaxidata/fastest_routes_train_part_1.csv/fastest_routes_train_part_1.csv')
train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
train_df = pd.read_csv('/Users/poojatyagi/Desktop/AI PROJECT/DATA SOURCES/train.csv')
train = pd.merge(train_df, train_fr_new, on = 'id', how = 'left')
train_df = train.copy()
train_df.head()

train_data_1 = train_df.copy()
if train_data_1.id.nunique() == train_data_1.shape[0]:
    print("Train ids are unique")
    
train_data_1.isnull().sum()
train_data_1 = train_data_1.dropna()

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_df['trip_duration'].values+1), axlabel = 'Log(trip_duration)', label = 'log(trip_duration)', bins = 50, color="r")
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(train_df['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(train_df['pickup_longitude'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])
sns.distplot(train_df['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(train_df['dropoff_longitude'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()



df = train_df.loc[(train_df.pickup_latitude > 40.6) & (train_df.pickup_latitude < 40.9)]
df = df.loc[(df.dropoff_latitude>40.6) & (df.dropoff_latitude < 40.9)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]

train_data_new = df.copy()

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(12, 12), sharex=False, sharey = False)#
sns.despine(left=True)
sns.distplot(train_data_new['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])
sns.distplot(train_data_new['pickup_longitude'].values, label = 'pickup_longitude',color="g",bins =100, ax=axes[0,1])
sns.distplot(train_data_new['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])
sns.distplot(train_data_new['dropoff_longitude'].values, label = 'dropoff_longitude',color="g",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()
print(df.shape[0], train_data.shape[0])
plt.show()


######################################################
temp = train_data_1.copy()
train_data_1['pickup_datetime'] = pd.to_datetime(train_data_1.pickup_datetime)
train_data_1.loc[:, 'pick_date'] = train_data_1['pickup_datetime'].dt.date
train_data_1.head()

ts_v1 = pd.DataFrame(train_data_1.loc[train_data_1['vendor_id']==1].groupby('pick_date')['trip_duration'].mean())
ts_v1.reset_index(inplace = True)
ts_v2 = pd.DataFrame(train_data_1.loc[train_data_1.vendor_id==2].groupby('pick_date')['trip_duration'].mean())
ts_v2.reset_index(inplace = True)

#from bokeh.sampledata.stocks import AAPL, IBM, MSFT, GOOG
output_notebook()
figg = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
figg.title.text = 'Click on legend entries to hide the corresponding lines'

for data, name, color in zip([ts_v1, ts_v2], ["vendor 1", "vendor 2"], Spectral4):
    df = data
    figg.line(df['pick_date'], df['trip_duration'], line_width=2, color=color, alpha=0.8, legend=name)

figg.legend.location = "top_left"
figg.legend.click_policy="hide"
show(figg)

train_data_1 = temp
###########################################3
rgb = np.zeros((3000, 3500, 3), dtype=np.uint8)
rgb[..., 0] = 0
rgb[..., 1] = 0
rgb[..., 2] = 0
train_data_new['pick_lat_new'] = list(map(int, (train_data_new['pickup_latitude'] - (40.6000))*10000))
train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))
train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))
train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))

summary_plot = pd.DataFrame(train_data_new.groupby(['pick_lat_new', 'pick_lon_new'])['id'].count())

summary_plot.reset_index(inplace = True)
summary_plot.head(120)
lat_list = summary_plot['pick_lat_new'].unique()
for i in lat_list:
    lon_list = summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()
    unit = summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()
    for j in lon_list:
        a = unit[lon_list.index(j)]
        if (a//50) >0:
            rgb[i][j][0] = 255
            rgb[i,j, 1] = 0
            rgb[i,j, 2] = 255
        elif (a//10)>0:
            rgb[i,j, 0] = 0
            rgb[i,j, 1] = 255
            rgb[i,j, 2] = 0
        else:
            rgb[i,j, 0] = 255
            rgb[i,j, 1] = 0
            rgb[i,j, 2] = 0
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,20))
ax.imshow(rgb, cmap = 'hot')
ax.set_axis_off() 

##################################################################################
'''
def color(hour):
    """function for color change in animation"""
    return(10*hour)

def Animation(hour, temp, rgb):
    """Function to generate return a pic of plotings"""
    #ax.clear()
    train_data_new = temp.loc[temp['hour'] == hour]
    #start = time.time()
    rgb = np.zeros((3000, 3500, 3), dtype=np.uint8)
    rgb[..., 0] = 0
    rgb[..., 1] = 0
    rgb[..., 2] = 0
    train_data_new['pick_lat_new'] = list(map(int, (train_data_new['pickup_latitude'] - (40.6000))*10000))
    train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))
    train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))
    train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))

    summary_plot = pd.DataFrame(train_data_new.groupby(['pick_lat_new', 'pick_lon_new'])['id'].count())

    summary_plot.reset_index(inplace = True)
    summary_plot.head(120)
    lat_list = summary_plot['pick_lat_new'].unique()
    for i in lat_list:
        #print(i)
        lon_list = summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()
        unit = summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()
        for j in lon_list:
            #j = int(j)
            a = unit[lon_list.index(j)]
            #print(a)
            if (a//50) >0:
                rgb[i][j][0] = 255 - color(hour)
                rgb[i,j, 1] = 255 - color(hour)
                rgb[i,j, 2] = 0 + color(hour)
            elif (a//10)>0:
                rgb[i,j, 0] = 0 + color(hour)
                rgb[i,j, 1] = 255 - color(hour)
                rgb[i,j, 2] = 0 + color(hour)
            else:
                rgb[i,j, 0] = 255 - color(hour)
                rgb[i,j, 1] = 0 + color(hour)
                rgb[i,j, 2] = 0 + color(hour)
    #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,20))
    #end = time.time()
    #print("Time taken by above cell is {} for {}.".format((end-start), hour))
    return(rgb)
'''    
    
    
  

##########################      FEATURE ENGINEERING        ##################################
#### FEATURE SELECTION - using backward propagation ####
print("FEATURE SELECTION \n")
'''Feature selection: we select a subset of the original feature set based on the statistical 
significance of different parameters.'''
#We will use backward elimination technique to select the best features to train our model.

#Let's assign the values to X & Y array from the dataset.
#First cheCK the index of the features and label
list(zip( range(0,len(train_data.columns)),train_data.columns))

Y = train_data.iloc[:,10].values
X = train_data.iloc[:,range(15,61)].values
X.shape
Y.shape
'''•duration variable assigned to Y because that is the dependent variable.
   •features such as id, timestamp and weekday were not assigned to X array because they are of type object.
    And we need an array of float data type.'''

print("Let's append {} rows of 1's as the first column in the X array".format(X.shape[0]))    

X1 = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X1.shape
#########################################################################################################
'''Here we will take the level of significance as 0.05 i.e. 5% which means that 
we will reject feature from the list of array and re-run the model
till p value for all the features goes below .05 to find out the optimal combination for our model. '''
X_opt = X1[:,[0,1,3,4,6,7,8,9,10,11]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Fetch p values for each feature
p_Vals = regressor_OLS.pvalues

#define significance level for accepting the feature.
sig_Level = 0.05

#Loop to iterate over features and remove the feature with p value less than the sig_level
while max(p_Vals) > sig_Level:
    print("Probability values of each feature \n")
    print(p_Vals)
    X_opt = np.delete(X_opt, np.argmax(p_Vals), axis = 1)
    print("\n")
    print("Feature at index {} is removed \n".format(str(np.argmax(p_Vals))))
    print(str(X_opt.shape[1]-1) + " dimensions remaining now... \n")
    regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
    p_Vals = regressor_OLS.pvalues
    print("=================================================================\n")
    
#Print final summary
print("Final stat summary with optimal {} features".format(str(X_opt.shape[1]-1)))
regressor_OLS.summary()

'''OBSERVATION ==> Finally we have reached the combination of optimum features with each feature having p value < 0.05.'''

######################################### SPLITING DATASET ################################
#taking logarithmic value of trip duration
y_many_features = np.log1p(train_data['trip_duration'])

#Split raw data
X_train, X_test, y_train, y_test = train_test_split(X,y_many_features, random_state=4, test_size=0.2)

#Split data from the feature selection group
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_opt,y_many_features, random_state=4, test_size=0.2)

########################## TRAINING and TESTING LINEAR REGRESSION MODEL ########################

''' EXPERIMENT 1 '''
#Linear regressor for the raw data
regressor = LinearRegression() 
regressor.fit(X_train,y_train) 
y_pred = regressor.predict(X_test) 

''' EXPERIMENT 2 '''
#Linear regressor for the Feature selection group
regressor1 = LinearRegression() 
regressor1.fit(X_train_fs,y_train_fs) 
y_pred_fs = regressor1.predict(X_test_fs) 



###########################TRAINING and TESTING RANDOM FOREST MODEL #################################

''' EXPERIMENT 3 '''
#instantiate the object for the Random Forest Regressor with default params from raw data
regressor_rfraw = RandomForestRegressor(n_jobs=-1)
regressor_rfraw.fit(X_train,y_train)
y_pred_rfraw = regressor_rfraw.predict(X_test)


''' EXPERIMENT 4 '''
#instantiate the object for the Random Forest Regressor with default params for Feature Selection Group
regressor_rf = RandomForestRegressor(n_jobs=-1)
regressor_rf.fit(X_train_fs,y_train_fs)
y_pred_rf = regressor_rf.predict(X_test_fs)


''' EXPERIMENT 5 '''
# #instantiate the object for the Random Forest Regressor with tuned hyper parameters for Feature Selection Group
regressor_rf1 = RandomForestRegressor(n_estimators = 200,
                                      max_depth = 22,
                                      min_samples_split = 9,
                                      n_jobs=-1)
regressor_rf1.fit(X_train_fs,y_train_fs)
y_pred_rf1 = regressor_rf1.predict(X_test_fs)


############################# TRAINING and TESTING XGBoost Regrssion ###########################
''' EXPERIMENT 6 '''
xgb_1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb_1.fit(X_train,y_train)
predictions_raw = xgb_1.predict(X_test)


''' EXPERIMENT 7 '''
xgb_2 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb_2.fit(X_train_fs,y_train_fs)
predictions = xgb_2.predict(X_test_fs)

print("\n")

############################# EVALUATION METRICS ###############################################

#defining function for Mean Absolute percentage error
def mean_absolute_percentage_error(y_test, y_pred):
    
    y_test, y_pred = np.array(y_test),np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test))*100 

#defining function for Root Mean square error
def rmse(y_test, y_pred):
    return np.sqrt(((y_pred - y_test) ** 2).mean())
#################################################################################################

print("\n \nEvaluating LINEAR REGRESSION:\n")
print("EXPERIMENT 1:")
print("Experiment 1 \t RMSE: ", rmse(y_test,y_pred))
print("Experiment 1 \t MAPE(%): ",mean_absolute_percentage_error(y_test,y_pred))
print('Experiment 1 \t Variance score(R squared): %.2f' % metrics.r2_score(y_test, y_pred))
print("\n")    
print("EXPERIMENT 2:")
print("Experiment 2 \t RMSE: ", rmse(y_test_fs,y_pred_fs))
print("Experiment 2 \t MAPE(%) : ",mean_absolute_percentage_error(y_test_fs,y_pred_fs))
print('Experiment 2 \t Variance score(R squared): %.2f' % metrics.r2_score(y_test_fs,y_pred_fs))
print("\n")
print("\n \nEvaluating RANDOM FORESTS :\n")
print("EXPERIMENT 3:")
print("Experiment 3 \t RMSE: ", rmse(y_test,y_pred_rfraw))
print("Experiment 3 \t MAPE(%): ",mean_absolute_percentage_error(y_test,y_pred_rfraw))
print('Experiment 3 \t Variance score(R squared): %.2f' % metrics.r2_score(y_test, y_pred_rfraw))
print("\n")    
print("EXPERIMENT 4:")
print("Experiment 4 \t RMSE: ", rmse(y_test_fs,y_pred_rf))
print("Experiment 4 \t MAPE(%): ",mean_absolute_percentage_error(y_test_fs,y_pred_rf))
print('Experiment 4 \t Variance score(R squared): %.2f' % metrics.r2_score(y_test_fs,y_pred_rf))
print("\n")
print("EXPERIMENT 5:")
print("Experiment 5 \t RMSE: ", rmse(y_test_fs,y_pred_rf1))
print("Experiment 5 \t MAPE(%): ",mean_absolute_percentage_error(y_test_fs,y_pred_rf1))
print('Experiment 5 \t Variance score(R squared): %.2f' % metrics.r2_score(y_test_fs,y_pred_rf1))
print("\n")
print("\n \n Evaluating XGBoost Regression :\n")
print("EXPERIMENT 6:")
print("Experiment 6 \t RMSE: ", rmse(y_test,predictions_raw))
print("Experiment 6 \t MAPE(%): ",mean_absolute_percentage_error(y_test,predictions_raw))
print('Experiment 6 \t Variance score(R squared):',metrics.r2_score(y_test, predictions_raw))
print("\n")
print("EXPERIMENT 7:")
print("Experiment 7 \t RMSE: ", rmse(y_test_fs,predictions))
print("Experiment 7 \t MAPE(%): ",mean_absolute_percentage_error(y_test_fs,predictions))
print('Experiment 7 \t Variance score(R squared): %.2f' % metrics.r2_score(y_test_fs,predictions))

############################################################################################################
########### VISUALIZATION OF ACCURACY ###############

#DISPLAYING THE IMPROVED TABLE OF ACCURACY OF ALL MODELS AFTER HYPER PARAMETER OPTIMIZATION
Summary_table_1 = pd.DataFrame([rmse(y_test,y_pred),rmse(y_test_fs,y_pred_fs),
                                rmse(y_test,y_pred),rmse(y_test_fs,y_pred_fs),
                                rmse(y_test_fs,y_pred_rf1),rmse(y_test,predictions_raw),
                                rmse(y_test_fs,predictions) ],
                                index=['Experiment 1', 'Experiment 2', 'Experiment 3','Experiment 4','Experiment 5',
                                'Experiment 6','Experiment 7'], columns=['RMSE'])


print("Summary Table for RMSE")
Summary_table_1

#PLOTTING GRAPGH OF ACCURACY
plt.figure(figsize=(10,5))
sns.barplot(Summary_table_1.index,Summary_table_1['RMSE'])
plt.show()


Summary_table_2 = pd.DataFrame([mean_absolute_percentage_error(y_test,y_pred),mean_absolute_percentage_error(y_test_fs,y_pred_fs),
                                mean_absolute_percentage_error(y_test,y_pred),mean_absolute_percentage_error(y_test_fs,y_pred_fs),
                                mean_absolute_percentage_error(y_test_fs,y_pred_rf1),mean_absolute_percentage_error(y_test,predictions_raw),
                                mean_absolute_percentage_error(y_test_fs,predictions) ],
                                index=['Experiment 1', 'Experiment 2', 'Experiment 3','Experiment 4','Experiment 5',
                                'Experiment 6','Experiment 7'], columns=['MAPE (Mean Absolute Percentage Error %)'])
print("Summary Table for MAPE")
Summary_table_2

#PLOTTING GRAPGH OF ACCURACY
plt.figure(figsize=(10,5))
sns.barplot(Summary_table_1.index,Summary_table_2['MAPE (Mean Absolute Percentage Error %)'])
plt.show()

Summary_table_3 = pd.DataFrame([metrics.r2_score(y_test,y_pred),metrics.r2_score(y_test_fs,y_pred_fs),
                                metrics.r2_score(y_test,y_pred),metrics.r2_score(y_test_fs,y_pred_fs),
                                metrics.r2_score(y_test_fs,y_pred_rf1),metrics.r2_score(y_test,predictions_raw),
                                metrics.r2_score(y_test_fs,predictions) ],
                                index=['Experiment 1', 'Experiment 2', 'Experiment 3','Experiment 4','Experiment 5',
                                'Experiment 6','Experiment 7'], columns=['Variance (R squared)'])

print("Summary Table for R2 Score")
Summary_table_3

#PLOTTING GRAPGH OF ACCURACY
plt.figure(figsize=(10,5))
sns.barplot(Summary_table_3.index,Summary_table_3['Variance (R squared)'])
plt.show()

########################################################## END #######################################################################################


















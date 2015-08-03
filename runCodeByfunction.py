def runHotel(trainingdata, testingData,):

    import pandas as pd
    import numpy as np
    df = pd.read_json(trainingdata)
    dfT = pd.read_json(testingData)
    #remove rows with dirty data from main testing dataset
    dfT.special_request_made = dfT.special_request_made.convert_objects(convert_numeric=True)
    dfT = dfT[np.isfinite(dfT['special_request_made'])]
    dfT = dfT.reset_index(drop=True)
    #finishing cleaning data
    
    features_all = df[['time','is_phone_booking', 'checkin', 'checkout']]
    #data for main test data
    features_all_mainData = dfT[['time','is_phone_booking', 'checkin', 'checkout']]
    target_all_for_mainTest = dfT[['cancelled']]
    target_all_for_mainTest = target_all_for_mainTest.convert_objects(convert_numeric=True)
    m= target_all_for_mainTest
    features_all_mainData['checkin'] = features_all_mainData['checkin'].astype('datetime64[ns]')
    features_all_mainData['checkout'] = features_all_mainData['checkout'].astype('datetime64[ns]')
    
    features_all['checkin'] = features_all['checkin'].astype('datetime64[ns]')
    features_all['checkout'] = features_all['checkout'].astype('datetime64[ns]')

    #get all outcomes from trainig data
    target_all = df[['cancelled']]
    #get the difference in days
    features_all['days_booked'] = (features_all['checkout'] - features_all['checkin']).dt.days
    features_all.drop(features_all.columns[[2,3]], axis = 1, inplace=True)

    features_all_mainData['days_booked'] = (features_all_mainData['checkout'] - features_all_mainData['checkin']).dt.days
    features_all_mainData.drop(features_all_mainData.columns[[2,3]], axis = 1, inplace=True)

    #convert timestap to day of the week Mon = 0, Tues = 1 etc
    features_all['time'] = features_all['time'].astype('datetime64[ns]').dt.dayofweek
    features_all_mainData['time'] = features_all_mainData['time'].astype('datetime64[ns]').dt.dayofweek
    
    #add a time of day feature 
    features_all['hour'] = features_all['time'].astype('datetime64[ns]').dt.hour
    features_all_mainData['hour'] = features_all_mainData['time'].astype('datetime64[ns]').dt.hour

    #features_all['hotel_state'] = features_all['hotel_state'].map({'Abia':1, 'Abuja':2, 'Adamawa':3})
    #features_all['hotel_state'] = features_all['hotel_state'].map({'Abia':1,'Abuja':2,'Adamawa':3,'AkwaIbom':4,'Anambra':5,'Bauchi':6,'Bayelsa':7,'Benue':8,'Borno':9,'Cross River':10,'Delta':11,'Ebonyi':12,'Edo':13,'Ekiti':14,'Enugu':15,'Gombe':16,'Imo':17,'Jigawa':18,'Kaduna':19,'Kano':20,'Katsina':21,'Kebbi':22,'Kogi':23,'Kwara':24,'Lagos':25,'Nassarawa':26,'Niger':27,'Ogun':28,'Ondo':29,'Osun':30,'Oyo':31,'Plateau':32,'Rivers':33,'Sokoto':34,'Taraba':35,'Yobe':36,'Zamfara':37features_all['hotel_state'] = features_all['hotel_state'].map({'Abia':1,'Abuja':2'Adamawa':3,'AkwaIbom':4,'Anambra':5,'Bauchi':6,'Bayelsa':7,'Benue':8,'Borno':9,'Cross River':10,'Delta':11,'Ebonyi':12,'Edo':13,'Ekiti':14,'Enugu':15,'Gombe':16,'Imo':17,'Jigawa':18,'Kaduna':19,'Kano':20,'Katsina':21,'Kebbi':22,'Kogi':23,'Kwara':24,'Lagos':25,'Nassarawa':26,'Niger':27,'Ogun':28,'Ondo':29,'Osun':30,'Oyo':31,'Plateau':32,'Rivers':33,'Sokoto':34,'Taraba':35,'Yobe':36,'Zamfara':37})
    #
    
    #split the outcomes into 4000 for train and 1000 for test
    target_train = target_all[:-1000]
    target_test = target_all[-1000:]
    #split the features into 4000 for train and 1000 for test
    features_train = features_all[:-1000]
    features_test = features_all[-1000:]
    
    #the learnig model used
    from sklearn import svm, linear_model
    clf1 = svm.SVC(kernel='rbf')
    clf1.fit(features_train, target_train)
    
    #another 
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(features_train, target_train)
    
    #predict the first xxx data from the main dataset
    result = clf1.predict(features_all_mainData[:])
    #a = logistic.predict(features_all_mainData[:])

    b = pd.DataFrame(result)
    b.columns = ['cancelled']
    #pedict for testing data
    A = clf1.predict(features_test)
    B = pd.DataFrame(A)
    B.columns = ['cancelled']
    #save it to a file called result. You can open it using excel
    #b.to_csv('result.csv', sep=',', encoding='utf-8')
    #target_test[-1000:].to_csv('real1000.csv', sep=',', encoding='utf-8')

    b = pd.DataFrame(result)
    b.columns = ['cancelled']
    #number_of_actual_cancelled = target_test[(target_test.cancelled == 1)]
    number_of_actual_cancelled = m[(m.cancelled == 1)]
    #number_of_predicted_cancelled = B[(B.cancelled == 1)]
    number_of_predicted_cancelled = b[(b.cancelled == 1)]
    
    f = '####################################################################################' + '\n' 
    f = f + 'RESULTS OF LEARNING AND PREDICTION' + '\n'
    f = f + '#######################################################################################' + '\n'
    s = ' Actual cancelled bookings in this dataset = '  + str(number_of_actual_cancelled.index.size) 
    z = 'Predicted cancelled outcome = ' + str(number_of_predicted_cancelled.index.size)
    n = ' Number of main testing dataset = ' + str(features_all_mainData.index.size)
    return f + n + "\n" + s +"\n " + z  + "\n " 



print runHotel('hotel_all.json', 'new_hotel.json')

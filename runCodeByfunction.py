def runHotel(trainingdata, testingData):

    import pandas as pd
    import numpy as np
    df = pd.read_csv(trainingdata)
    dfT = pd.read_csv(testingData)
    features_all = df[['time','is_phone_booking','number_of_rooms', 'client_title']]
    #data for main test data
    features_all_mainData = dfT[['time','is_phone_booking','number_of_rooms', 'client_title']]
    #get all outcomes from trainig data
    target_all = df[['cancelled']]
    #target_all_for_mainTest = dfT[['cancelled']]
    #convert timestap to day of the week Mon = 0, Tues = 1 etc
    features_all['time'] = features_all['time'].astype('datetime64[ns]').dt.dayofweek
    features_all_mainData['time'] = features_all_mainData['time'].astype('datetime64[ns]').dt.dayofweek
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
    clf = svm.SVC(kernel='linear')
    clf.fit(features_train, target_train)
    
    #predict the first xxx data from the main dataset
    a = clf.predict(features_all_mainData[-3000:])
    
    b = pd.DataFrame(a)
    #save it to a file called result. You can open it using excel
    b.to_csv('result.csv', sep=',', encoding='utf-8')
    target_test[-1000:].to_csv('real1000.csv', sep=',', encoding='utf-8')
    return  b

runHotel('hotel_all.csv', 'hotel_all.csv' )

import pandas as pd
import pandas as pd
import numpy as np
df = pd.read_csv('hotel_all.csv')
features_all = df[['time','is_phone_booking','number_of_rooms']]
target_all = df[['cancelled']]
#features_all['client_title'] = features_all['client_title'].map({'Mr.':1., 'Mrs.':0., 'Miss.':0. })
features_all['time'] = features_all['time'].astype('datetime64[ns]').dt.dayofweek
#features_all['hotel_state'] = features_all['hotel_state'].map({'Abia':1, 'Abuja':2, 'Adamawa':3})
#features_all['hotel_state'] = features_all['hotel_state'].map({'Abia':1,'Abuja':2,'Adamawa':3,'AkwaIbom':4,'Anambra':5,'Bauchi':6,'Bayelsa':7,'Benue':8,'Borno':9,'Cross River':10,'Delta':11,'Ebonyi':12,'Edo':13,'Ekiti':14,'Enugu':15,'Gombe':16,'Imo':17,'Jigawa':18,'Kaduna':19,'Kano':20,'Katsina':21,'Kebbi':22,'Kogi':23,'Kwara':24,'Lagos':25,'Nassarawa':26,'Niger':27,'Ogun':28,'Ondo':29,'Osun':30,'Oyo':31,'Plateau':32,'Rivers':33,'Sokoto':34,'Taraba':35,'Yobe':36,'Zamfara':37features_all['hotel_state'] = features_all['hotel_state'].map({'Abia':1,'Abuja':2'Adamawa':3,'AkwaIbom':4,'Anambra':5,'Bauchi':6,'Bayelsa':7,'Benue':8,'Borno':9,'Cross River':10,'Delta':11,'Ebonyi':12,'Edo':13,'Ekiti':14,'Enugu':15,'Gombe':16,'Imo':17,'Jigawa':18,'Kaduna':19,'Kano':20,'Katsina':21,'Kebbi':22,'Kogi':23,'Kwara':24,'Lagos':25,'Nassarawa':26,'Niger':27,'Ogun':28,'Ondo':29,'Osun':30,'Oyo':31,'Plateau':32,'Rivers':33,'Sokoto':34,'Taraba':35,'Yobe':36,'Zamfara':37})
target_train = target_all[:-1000]
target_test = target_all[-1000:]
features_train = features_all[:-1000]
features_test = features_all[-1000:]
from sklearn import svm, linear_model
clf = svm.SVC(kernel='linear')
clf.fit(features_train, target_train)
clf.score(features_test,target_test)
clf.predict(features_test[-1000:])
a = clf.predict(features_test[-1000:])
b = pd.DataFrame(a)
b.to_csv('result.csv', sep=',', encoding='utf-8')
target_test[-1000:].to_csv('real1000.csv', sep=',', encoding='utf-8')

import os
import sys
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler

import xgboost as xgb



#load training sample
indict=  np.load('reg_tuples_2.py.npy',allow_pickle=True) # load
indict = np.array(indict)
indict = indict.tolist()
print(indict.keys())
#sns.pairplot(indict[['Muon_matched_pt', 'Muon_matched_eta', 'Jet_matched_pt', 'GenJet_matched_pt']], diag_kind='kde')
#print(indict.values())
#print(indict)

#pop out some interesting columns for future 2018 to bMuReg comparison - note: the popped columns are not included in the regression
reg2018 = indict['Jet_matched_bReg2018']
pTRaw= indict['Jet_matched_pTraw']

pt_true_start= indict.pop('GenJet_matched_pt')
pTReco= indict.pop('Jet_matched_pt')
eta_reco= indict['Jet_matched_eta']
mass= indict.pop('Jet_matched_mass')

pT_JEC_reg2018 = pTReco*reg2018
mask = pTReco>60 
mask_eta =abs(eta_reco)<2.4
pt_true_start =pt_true_start[mask & mask_eta] 
pT_JEC_reg2018 =pT_JEC_reg2018[mask & mask_eta] 
pTReco = pTReco[mask & mask_eta]
#print(reg2018)
#print(pTReco)
#print(pt_true_start)

#set label for which value to regress on
labels = indict.pop('Jet_GenJet_pTratio')

#apparently nMuons not saved as int - check and FIXME
pTReco= indict.pop('Jet_matched_nMuons') 

indict = indict.values()
indict = np.stack(list(indict), axis=-1)


#split in train and test
train_dataset, test_dataset,train_labels,test_labels  = train_test_split(indict,labels, test_size=0.2)


#define test and train features 
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#scale variables 
object= RobustScaler()
#normalizer = tf.keras.layers.Normalization(axis=-1)
X_train = object.fit_transform(train_features) 
X_test= object.transform(test_features) 

#regression definition
clf = xgb.XGBRegressor(objective='reg:squarederror')

#regression fit 
clf.fit(X_train,train_labels)

#predictions
predictions = clf.predict(X_test)
print(train_features[:,])
print(len(test_features[:,1]))
print(len(test_labels))
true = test_labels.ravel() #GenJet_pt/Jet_pt
recoPt = test_features[:,2].ravel()
reg2018 =test_features[:,3].ravel() 
print(reg2018[0:10])
ratio = true/predictions
#ratio=1/ratio
print('')
regressed_pt = predictions*recoPt #recoPt here is without JEC/JER (corrected with rawFactor - https://github.com/Raffaella07/BBParkingNano/blob/master/NanoAOD/python/jets_cff.py#L223)
true_pt = true*recoPt # Gen_pt
pT_reg2018 = recoPt*reg2018 # rawPt * bReg2018
print(pT_reg2018)
#_pt = true*recoPt
print(regressed_pt)
print(true_pt)
print(recoPt)

#reco_ratio=1./reco_ratio

print(ratio)

#compare different spectra 
bin_edges = np.linspace(0, 3, 100) 
print(pt_true_start/pT_JEC_reg2018)
plt.figure(figsize=(10, 6)) # Set the figure size (optional)
plt.hist(true_pt/recoPt, label='reco pt',bins=bin_edges, color='blue',histtype='step') # Plot y1
plt.hist(true_pt/regressed_pt, label='regressed',bins=bin_edges, color='red',histtype='step') # Plot y2
plt.hist(true_pt/pT_reg2018, label=' 2018 regression',bins=bin_edges, color='green',histtype='step') # Plot y2
plt.hist(pt_true_start/pT_JEC_reg2018, label=' 2018 regression, pt>60 GeV',bins=bin_edges, color='orange',histtype='step') # Plot y2
#plt.hist(regressed_pt, label='regressed pt ', bins = 50, color='green') # Plot y3

# Adding title and labels
plt.xlabel('p^{gen}_{T}/p_{T} (GeV)')
plt.ylabel('arbitrary units')

# Adding a legend
plt.legend()

# Display the plot
plt.savefig("../plots/test_regression.png")
plt.savefig("../plots/test_regression.pdf")
#initialize Keras model
#model = Sequential()
#
##add layers 
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(10))
#
#model.compile(optimizer='Adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#
#print(X_train.shape)
#print(train_labels.shape)
#
#model.fit(X_train, train_labels,
#          batch_size=128,
#          epochs=2,
#          verbose=1,
#          validation_data=(X_test, test_labels))
#
#
#
#score = model.evaluate(X_test, test_labels, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#
#y_pred = model.predict(X_test)
#print(y_pred)

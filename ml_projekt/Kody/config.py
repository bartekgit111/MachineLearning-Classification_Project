import sklearn.model_selection as skm

nr_index = 327426

cv = 7

kfold = skm.KFold(cv, random_state=nr_index, shuffle=True)
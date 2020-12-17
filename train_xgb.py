import numpy as np
import pandas as pd
from keras.models import load_model
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from eval import GetTrainingSet
from keras.models import Model
import pickle
import h5py
import sklearn
from keras.utils import np_utils
from functools import partial
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt.space import Integer, Real
from sklearn.model_selection import PredefinedSplit
from sklearn.tree import DecisionTreeClassifier
from args import parse_args
from sklearn.externals.joblib import parallel_backend
from sklearn import metrics

class StackedModel:

    def __init__(self, name_cnn, name_boost):

        self.name_cnn = name_cnn
        self.name_boost = name_boost
        self.args = parse_args()

    def Train(self, known_maps = False):

        args = self.args
        traindir = args.traindir + '/data_train.h5'
        testdir = args.testdir + '/data_test.h5'
        #traindir = testdir
        validdir = args.validdir + '/data_valid.h5'
    
        print("... getting data ...") 
        _,_, y = GetTrainingSet(traindir)
        train_ind = np.full((y.shape[-1],),-1, dtype = int)

        _, _, y_valid = GetTrainingSet(validdir) 
        test_ind = np.full((y_valid.shape[-1],),0, dtype = int)    
        #eval_set = [([X1_valid, X2_valid],y_valid)]
 
        y = np.concatenate((y, y_valid))
        test_fold = np.append(train_ind, test_ind)

        lb_xgb = np.argmax(y, axis=1)
        
        if known_maps == False:

            X1, X2, y = GetTrainingSet(traindir)
            X1_valid,X2_valid, y_valid = GetTrainingSet(validdir)
            X1 = np.concatenate((X1, X1_valid), axis=0) 
            X2 = np.concatenate((X2, X2_valid), axis=0)
            print("... loading {0} model ...".format(self.name_cnn))  
            model = load_model("Models/{0}.h5".format(self.name_cnn))
            print(model.summary())

            model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)
            #model_feat = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)
            print("... getting predicted features ...")  
            feat_train = model_feat.predict([X1,X2], verbose = 1)
        
            print("... saving predictions ... ")
            hf = h5py.File('Predictions/ResNet128/train_feature_maps_{n}.h5'.format(n=self.name_cnn), 'w')
            hf.create_dataset('feat_train', data=feat_train)
            hf.close()
        else:
            hf = h5py.File('Predictions/ResNet128/train_feature_maps_{n}.h5'.format(n=self.name_cnn), 'r')
            feat_train= np.array(hf.get('feat_train'))
            print("... Feature maps input shape:    ", feat_train.shape)

        print("... fitting {0} model ...".format(self.name_boost))

        if self.name_boost=="XGB":  
            #model = xgb.XGBClassifier(n_estimators = 150, learning_rate = 0.1, max_depth = 12)
            model = xgb.XGBClassifier(n_estimators = 150, learning_rate = 0.1, max_depth = 10)
            param = {'max_depth': np.arange(4, 20),
                 'min_child_weight': [0.5, 1, 5, 10, 30],
                 'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
                 'n_estimators': [20, 40, 60, 100, 150, 200]}
        elif self.name_boost=="AdaBoost":
            #model = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.2, base_estimator = DecisionTreeClassifier(max_depth = 6))
            model = AdaBoostClassifier(n_estimators = 300, learning_rate = 0.05, base_estimator = DecisionTreeClassifier(max_depth = 8))
            param = {'base_estimator': [DecisionTreeClassifier(max_depth = 2), DecisionTreeClassifier(max_depth = 3), DecisionTreeClassifier(max_depth = 4)],
                     'n_estimators': [20, 40, 60, 100, 150, 200],
                     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]}
        elif self.name_boost=="GradientBoosting":
            model = GradientBoostingClassifier()
            param = {'n_estimators': np.arange(20, 110, 5),
                     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                     'max_depth': np.arange(4, 20)}
        elif self.name_boost=="RandomForest":
            #model = RandomForestClassifier(n_estimators = 500, max_depth = 18)
            model = RandomForestClassifier(n_estimators = 300, max_depth = 14)
            param = {'n_estimators': [20, 40, 60, 100, 150, 200],
                     'max_depth': np.arange(4, 20)}

        ps = PredefinedSplit(test_fold) 


        scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True, labels=sorted(np.unique(lb_xgb)))
        #opt = BayesSearchCV(model, param, n_iter=10,random_state=42, cv = 10, verbose = 10, n_jobs = -1, scoring = 'accuracy') #n_iter=100,random_state=42, 
        print("... fitting Bayes Search ...")
        with parallel_backend('threading'):
            #opt.fit(feat_train,lb_xgb)
            model.fit(feat_train,lb_xgb)
        #print(".....Best params ...", opt.best_params_)
        with open("eval.txt", "a") as text_file:
            text_file.write("{m}_{n} \n".format(m=self.name_cnn,n=self.name_boost))
            #text_file.write("best params: {0} \n".format(opt.best_params_))
            #text_file.write("feature importances: {0} \n".format(feat_importance))
        print("Evaluation done.")

        # save the best model
        #model= opt.best_estimator_
        pickle.dump(model, open("Models/{m}_{n}.h5".format(m=self.name_boost,n=self.name_cnn), "wb"))
        print("training accuracy: ... ",model.score(feat_train,lb_xgb))


    def Predict(self,path_data):

        path_model = 'Models/ResNet128/{0}.h5'.format(self.name_cnn)
        print("... loading model from ... ", path_model)
        model = load_model(path_model)
        model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)

        path_boost = "Models/{m}_{n}.h5".format(m=self.name_boost,n=self.name_cnn)
        print("... loading model from ... ", path_boost)
        model = pickle.load(open(path_boost, 'rb'))
   
        print("... getting data ...") 
        X1, X2, y = GetTrainingSet(path_data) 

        print("... getting predicted features ...") 
        feat_val = model_feat.predict([X1, X2], verbose=1)
        print("... calculating predictions ... ")
        y_pred = model.predict_proba(feat_val)
        print(y_pred)

        print("... saving predictions ... ")
        hf = h5py.File('Predictions/{m}_{n}_{p}.h5'.format(m=self.name_boost,n=self.name_cnn,p=str(path_data[-12:-3])), 'w')
        hf.create_dataset('y_pred', data=y_pred)
        hf.create_dataset('y_true', data=y)
        hf.close()
   



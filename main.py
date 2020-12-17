from eval import Evaluate, ControlPlots
from train_xgb import StackedModel
from train import ModelCNN

data_test = "/mnt/lustre/helios-home/kubumiro/dl/TensorFlow-GPU/Masters/Masters/Data_split/data_test/data_test.h5"
data_valid = "/mnt/lustre/helios-home/kubumiro/dl/TensorFlow-GPU/Masters/Masters/Data_split/data_valid/data_valid.h5"
data_train = "/mnt/lustre/helios-home/kubumiro/dl/TensorFlow-GPU/Masters/Masters/Data_split/data_train/data_train.h5"

# model names ... "CNN", "ResNet", "ShortCNN" and "XGB","AdaBoost", GradientBoosting

#model = ModelCNN(name = "ShortCNN")
#model.Train()
#model.Predict(data_valid)

#Evaluate("CVN",data_test, histo = True)


model = StackedModel(name_cnn = "ResNet",name_boost = "XGB")
#model.Train(known_maps = True)
model.Predict(data_train)
Evaluate("XGB_ResNet",data_train, histo = True)

model = StackedModel(name_cnn = "ResNet",name_boost = "RandomForest")
model.Predict(data_train)
Evaluate("RandomForest_ResNet",data_train, histo = True)

model = StackedModel(name_cnn = "ResNet",name_boost = "AdaBoost")
model.Predict(data_train)
Evaluate("AdaBoost_ResNet",data_train, histo = True)



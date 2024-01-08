import pandas as pd
import numpy as np
import os
import pickle , joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,train_test_split
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor





# # reading data form csv file
path = os.path.abspath("car.csv")
data_df = pd.read_csv(path)



# # hear we are getting or understanding some things 
# print(data_df.shape)
# print(data_df.describe())

# # converting data into numpy array or can skip this step
data_num = data_df.values
# print(data_num)


# # now we are trying to train test split the data # if you skip above step then conver from pandas it is vert esy
x = data_df.drop(columns=["Selling_Price","Owner", "Transmission", "Seller_Type", "Fuel_Type", "Car_Name"]).values
# print(x[:1, :])
# print(type(x))

y = data_df["Selling_Price"].values
y = y.reshape(-1,1)
# print(y[:1,:])
# print(type(y))


# # now we are drowing the Scatters on this data
# plt.plot(x[:,1:2],y,marker='o', linestyle='')
scatter_matrix(data_df, figsize = (12,8))
# plt.show()


# # now finding correlation with y 
corr = data_df[["Kms_Driven", "Present_Price", "Year"] + ["Selling_Price"]].corr()
# print(corr)




# # now we are going to split the data into train and test
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.20)




# # now we are test this model in diffrant models
# 1.linear_model
my_linear_model = linear_model.LinearRegression()
my_linear_model.fit(x_train, y_train)
my_linear_model_y_pred = my_linear_model.predict(x_test[:])

print("Orignal price is : ", y_test[:])
print("Predicted model is : ", my_linear_model_y_pred)
# print("accuracy score is : ", accuracy_score(y_test[:1], my_linear_model_y_pred[:1]))
print("mean squared error is : ", mean_squared_error(y_test[:], my_linear_model_y_pred))




# # # # 2.Disition Tree
# my_disition_tree = DecisionTreeRegressor()
# my_disition_tree.fit(x_train, y_train)
# my_disition_tree_y_pred = my_disition_tree.predict(x_test[:])

# print("Orignal price is : ", y_test[:])
# print("Predicted model is : ", my_disition_tree_y_pred)
# # print("accuracy score is : ", accuracy_score(y_test[:1], my_disition_tree_y_pred))
# print("mean squared error is : ", mean_squared_error(y_test[:], my_disition_tree_y_pred))




# # 3.Randome Foreste
# my_rand_fore = RandomForestRegressor()
# my_rand_fore.fit(x_train, y_train)
# my_disition_tree_y_pred = my_rand_fore.predict(x_test[:])

# print("Orignal price is : ", y_test[:])
# print("Predicted model is : ", my_disition_tree_y_pred)
# # print("accuracy score is : ", accuracy_score(y_test[:], my_disition_tree_y_pred))
# print("mean squared error is : ", mean_squared_error(y_test[:], my_disition_tree_y_pred))



# # we are doing a Creoss validation hear
try:
    cro_vali_scor = cross_val_score(my_linear_model, x_test, y_test, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-cro_vali_scor)
    print("rmse scores : ", rmse_scores)
    print("rmse scores mean : ", rmse_scores.mean())
    print("rmse scores std : ", rmse_scores.std())
except Exception as e:
    print(e)






# # List of model parformance
    
# 1. Linear Regressor model
# mean squared error is :  4.010554030281882
# cross validation :    mean = 1.91921145696511
#                       std  = 0.9345995204741182



# 2.Disition Tree
# mean squared error is :  1.9466852459016395
# cross validation :    mean = 2.7057840330023746
#                       std  = 2.381146713562881


# 3.Random Forest
# mean squared error is :  0.9005912544262308
# fine tuning
    





# # now from this outcomes we desided that our best model is a Linear Regressor model beacuse Random Forest do overfiting and Disition Tree is not too much good numbers.
# so now exporting our model

# Save the model to a file
model_filename = 'Machine_Learning_series/pro_1_car_prediction/linear_regression_model.joblib'
joblib.dump(my_linear_model, model_filename)

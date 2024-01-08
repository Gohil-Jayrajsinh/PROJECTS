 doing a Creoss validation hear
# try:
#     cro_vali_scor = cross_val_score(my_rand_fore, x_test, y_test, scoring="neg_mean_squared_error", cv=10)
#     rmse_scores = np.sqrt(-cro_vali_scor)
#     print("rmse scores : ", rmse_scores)
#     print("rmse scores mean : ", rmse_scores.mean())
#     print("rmse scores std : ", rmse_scores.std())
# except Exception as e:
#     print(e)
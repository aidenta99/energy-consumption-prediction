import feature_engineering

# create dummy variables from categorical variables
data = pd.get_dummies(data).reset_index(drop=True)

target_column = "energy consumption"

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=target_column), data[target_column]
)

X_train = X_train.drop('Sample Time 1', axis='columns')
X_test = X_test.drop('Sample Time 1', axis='columns')

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Linear Regression
# Lasso Regression
# Logistic Regression
# Random forest
# Neural network

# Baseline model: Linear Regression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Lasso Regression
# Hyper-parameter tuning for Lasso Regression
lm_l = Lasso(tol=0.01, alpha=0.13)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Tuning the model using GridSearch
# lasso_parameters = {'alpha':[i/100 for i in range(1,100)]}
# gs_lasso = GridSearchCV(lm_l, lasso_parameters, scoring='neg_mean_absolute_error')
# gs_lasso.fit(X_train, y_train)
# gs_lasso.best_score_
# gs_lasso.best_estimator_

# Ridge Regression
lm_r = Ridge()
lm_r.fit(X_train, y_train)
np.mean(cross_val_score(lm_r, X_train, y_train, scoring = 'neg_mean_absolute_error'))

# Random Forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error'))

parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

# Tuning random forest using GridSearchCv
gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)
gs.best_score_
gs.best_estimator_

from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1),max_features=1.0)
iso.fit(X_train)
ypred_iso = iso.fit_predict(X_train)
ypred_iso = pd.DataFrame(ypred_iso)
ypred_iso[0].value_counts()

# Test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lasso = lm_l.predict(X_test)
tpred_ridge = lm_r.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

print("MAE of LR: ", mean_absolute_error(y_test, tpred_lm))
print("MAE of Lasso: ", mean_absolute_error(y_test, tpred_lasso))
print("MAE of Ridge: ", mean_absolute_error(y_test, tpred_ridge))
print("MAE of Random Forest: ", mean_absolute_error(y_test, tpred_rf))

print("R2 of LR: ", r2_score(y_test, tpred_lm))
print("R2 of Lasso: ", r2_score(y_test, tpred_lasso))
print("R2 of Ridge: ", r2_score(y_test, tpred_ridge))
print("R2 of Random Forest: ", r2_score(y_test, tpred_rf))


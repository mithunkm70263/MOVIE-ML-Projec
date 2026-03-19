import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
ImDB_data = "/Users/mithunkm/Desktop/First Ml Project(me)/movie_data.csv.csv"
def load_csv_data(ImDB_data):
    imdb_csv = pd.read_csv(ImDB_data)
    return imdb_csv
movie_df = load_csv_data(ImDB_data)

#clean data
movie_df["age_certification"]=movie_df["age_certification"].fillna("Unknown")
movie_df["imdb_votes"]=movie_df["imdb_votes"].fillna(movie_df["imdb_votes"].median())
movie_df=movie_df.drop(["index","id","title","description","imdb_id"],axis=1)

#feature engineering
movie_df= pd.get_dummies(movie_df,columns=["type","age_certification"])
movie_df["imdb_votes"] = np.log1p(movie_df["imdb_votes"])
y = movie_df["imdb_score"]
X =movie_df.drop("imdb_score",axis=1)

#train,test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
coefficients = pd.Series(lr_model.coef_, index=X.columns)
print(coefficients.sort_values())

print("MSE: ",mean_squared_error(y_test,y_pred))
print("MAE: ",mean_absolute_error(y_test,y_pred))
print("r2_score: ",r2_score(y_test,y_pred))


residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0)  # horizontal line at 0

plt.xlabel("Predicted Values")
plt.ylabel("Residuals (y_test - y_pred)")
plt.title("Residual Plot")
# Train Linear Regression separately for interpretation
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
coefficients = pd.Series(lr_model.coef_, index=X.columns)
print(coefficients.sort_values())
plt.show()

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.show()


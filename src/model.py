"""module importtations"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

"""defining the fucntion for train model"""

def train_model(df):
    features = df[["Hours","Attendance","Sleep_Hours","Practice_Problems"]]
    target = df["Scores"]

    features_train,features_test,target_train,target_test = train_test_split(
        features,target, test_size=0.2, random_state=42
        )

    model = LinearRegression()
    model.fit(features_train,target_train)

    return model, features_test, target_test

def evaluate_model(model,features_test,target_test):
    """defining the function for evaluate the model"""

    predictions = model.predict(features_test)
    mse = mean_squared_error(target_test,predictions)

    return mse

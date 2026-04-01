# regression_models.py

def get_regression_model(model_name, X_train, y_train):

    model_name = model_name.strip().lower()

    if model_name == "linear_regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    elif model_name == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge()

    elif model_name == "lasso":
        from sklearn.linear_model import Lasso
        model = Lasso()

    elif model_name == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()

    elif model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor()

    elif model_name == "svr":
        from sklearn.svm import SVR
        model = SVR()

    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )

    else:
        raise ValueError("Model not found")

    model.fit(X_train, y_train)
    return model
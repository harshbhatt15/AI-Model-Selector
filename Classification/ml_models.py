def get_model(model_name, X_train, y_train):

    model_name = model_name.strip().lower() 

    if model_name == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    
    elif model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    
    elif model_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    
    elif model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
    
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    else:
        print("ERROR MODEL NAME:", repr(model_name))  # debug
        raise ValueError("Model not found")

    model.fit(X_train, y_train)
    return model
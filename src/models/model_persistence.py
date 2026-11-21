def save_sarima_model(model, file_path):

    # SARIMA modelini kaydediyoruz.
    model.save(file_path)


def save_xgboost_model(model, file_path):

    # XGBoost modelini kaydediyoruz.
    model.save_model(file_path)


def save_seasonal_naive_model(model, file_path):

    # Son 12 aylık satış değerlerini kaydediyoruz.
    model.to_csv(file_path)


def save_selected_model(model, model_name, file_path):

    # Modelin türüne göre kaydetme işlemini gerçekleştiriyoruz.
    if model_name == "SARIMA":
        save_sarima_model(model, file_path)

    elif model_name == "XGBoost":
        save_xgboost_model(model, file_path)

    elif model_name == "Seasonal Naive":
        save_seasonal_naive_model(model, file_path)

    else:
        raise ValueError(f"Unknown model: {model_name}")
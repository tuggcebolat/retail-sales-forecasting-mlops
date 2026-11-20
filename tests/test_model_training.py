from src.pipelines.data_pipeline import prepare_sales_data
from src.models.model_training import train_selected_model

# Prepare historical sales data.
df = prepare_sales_data()

# Train the selected model.
model = train_selected_model(df, "SARIMA")

# Display the trained model.
print(type(model))

# Train the XGBoost model.
model = train_selected_model(df, "XGBoost")

# Display the trained model type.
print(type(model))

# Prepare the Seasonal Naive model.
model = train_selected_model(df, "Seasonal Naive")

# Verify that 12 monthly observations are stored.
assert len(model) == 12

print(model)
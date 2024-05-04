from fbprophet import Prophet

# Train Prophet model for each category separately
prophet_models = {}
for category in df.columns[1:]:
    df_prophet = df[[category]].reset_index().rename(columns={'datetime':'ds', category:'y'})
    model = Prophet()
    model.fit(df_prophet)
    prophet_models[category] = model

# Forecast using Prophet model
prophet_forecasts = {}
for category, model in prophet_models.items():
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    prophet_forecasts[category] = forecast[['ds', 'yhat']].set_index('ds').rename(columns={'yhat':category})

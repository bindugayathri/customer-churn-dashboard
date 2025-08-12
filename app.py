# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go

# --- 1. Load Artifacts and Data ---
# Load the trained model and the preprocessor
model = joblib.load('./artifacts/churn_model.pkl')
preprocessor = joblib.load('./artifacts/preprocessor.pkl')

# Load the raw data to get feature values for a given customerID
raw_df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Keep a copy of the customer IDs for later lookup
customer_ids = raw_df['customerID']
# Drop the customerID for processing
raw_df_for_processing = raw_df.drop(columns=['customerID', 'Churn'])

# Create a SHAP explainer object for our tree-based model (XGBoost)
explainer = shap.TreeExplainer(model)

# --- 2. Initialize the Dash App ---
app = dash.Dash(__name__)
server = app.server # Expose the server for Gunicorn

# --- 3. Define the App Layout ---
app.layout = html.Div([
    # Header
    html.H1("Customer Churn Prediction Dashboard", style={'textAlign': 'center'}),
    html.P("Enter a customerID to predict their churn risk and see the key factors influencing the prediction.", style={'textAlign': 'center'}),
    
    # Input section
    html.Div([
        dcc.Input(
            id='customer-id-input',
            type='text',
            placeholder='Enter customerID (e.g., 7590-VHVEG)',
            style={'width': '300px', 'marginRight': '10px'}
        ),
        html.Button('Predict', id='predict-button', n_clicks=0),
    ], style={'textAlign': 'center', 'padding': '20px'}),
    
    # Output section
    html.Div(id='prediction-output', style={'padding': '20px', 'width': '80%', 'margin': 'auto'}),
    
], style={'fontFamily': 'Arial'})

# --- 4. Define Callbacks for Interactivity ---
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('customer-id-input', 'value')]
)
def update_prediction(n_clicks, customer_id):
    # Only run if the button has been clicked
    if n_clicks == 0 or not customer_id:
        return ""

    # Check if the customerID exists in our dataset
    if customer_id not in customer_ids.values:
        return html.Div(f"Error: customerID '{customer_id}' not found.", style={'color': 'red', 'textAlign': 'center'})

    # Get the single row of data for the customer
    customer_data = raw_df_for_processing[customer_ids == customer_id]
    
    # Preprocess the customer's data using the loaded preprocessor
    customer_processed = preprocessor.transform(customer_data)
    
    # Make a prediction and get the probability
    prediction = model.predict(customer_processed)[0]
    probability = model.predict_proba(customer_processed)[0][1] # Probability of churn
    
    # Get SHAP values for the single prediction
    shap_values = explainer.shap_values(customer_processed)
    
    # Get the feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # --- Create the output display ---
    prediction_text = "YES" if prediction == 1 else "NO"
    churn_risk_style = {'color': 'red', 'fontWeight': 'bold'} if prediction == 1 else {'color': 'green', 'fontWeight': 'bold'}

    # Create the SHAP Waterfall Plot
    waterfall_fig = go.Figure(go.Waterfall(
        name = "Prediction",
        orientation = "h",
        measure = ["relative"] * len(shap_values[0]),
        y = feature_names,
        x = shap_values[0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        base = explainer.expected_value
    ))
    waterfall_fig.update_layout(
        title='How Features Influence the Churn Prediction',
        showlegend=False,
        yaxis_title="Features",
        xaxis_title="Contribution to Churn Prediction"
    )
    
    return html.Div([
        html.H3(f"Churn Risk Prediction: ", style={'textAlign': 'center'}),
        html.H2(f"{prediction_text} (Risk Score: {probability:.0%})", style={**churn_risk_style, 'textAlign': 'center'}),
        dcc.Graph(figure=waterfall_fig)
    ])

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
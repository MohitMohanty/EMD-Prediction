import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load your Excel data
file_path = "AI_TABLE.xls"
df = pd.read_excel(file_path)

df = df.head(30)
df.fillna("unassigned", inplace=True) 

# Assume you want to predict the "target_column" based on the "feature_columns" and "categorical_columns"
text_feature_columns = ["DEFECT_DESC", "JOBDETAIL", "JOBSUM", "JOBHEAD"]
categorical_columns = ["STR_SHIPNAME", "STR_REFIT_CODE", "FLG_OFFLOADED", "EQUIPMENT_NAME", "WI_QC_REMARK"]
target_column = "EMD"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[text_feature_columns + categorical_columns], df[target_column], test_size=0.2, random_state=42
)

# Use CountVectorizer to convert text data to numeric format for text columns
text_vectorizer = ColumnTransformer(
    transformers=[
        ('defect_text', CountVectorizer(), 'DEFECT_DESC'),
        ('jobdetail_text', CountVectorizer(), 'JOBDETAIL'),
        ('jobsum_text', CountVectorizer(), 'JOBSUM'),
        ('jobhead_text', CountVectorizer(), 'JOBHEAD')
    ],
    remainder='passthrough'
)

categorical_encoder = OneHotEncoder()

# Combine text and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_vectorizer, text_feature_columns),
        ('categorical', categorical_encoder, categorical_columns)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and the RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model
model.fit(X_train, y_train)

# Create Dash app
app = Dash(__name__)

app.title = "EMD - PREDICTION"

app.layout = html.Div([
    html.H1('Predict Man Days', style={'textAlign': 'center', 'color': '#5072A7', 'fontFamily': 'Arial', 'fontWeight': 'bold', 'padding': '20px'}),

    html.Div([
        html.Table([
            # Row 1
            html.Tr([
                html.Td([html.Label('Defect Description:', style={'margin': '10px'}),]),
                html.Td([
                    dcc.Input(id='defect-desc-input', type='text', placeholder='Enter defect description', style={'width': '300px', 'margin': '10px', 'margin-right': '50px'})
                ]),
                html.Td([html.Label('Select Shipname:', style={'margin': '10px'}), ]),
                html.Td([
                    dcc.Dropdown(
                        id='categorical-dropdown-shipname',
                        options=[{'label': col, 'value': col} for col in df["STR_SHIPNAME"].unique()],
                        value=df["STR_SHIPNAME"].unique()[0],
                        placeholder='Select a ship name',
                        style={'width': '250px', 'margin': '10px', 'margin-right': '50px'}
                    )
                ]),
                html.Td([html.Label('Select QC Remarks :', style={'width': '150px' , 'margin': '10px'}),]),
                html.Td([
                    dcc.Dropdown(
                        id='categorical-dropdown-qcremark',
                        options=[{'label': col, 'value': col} for col in df["WI_QC_REMARK"].unique()],
                        value=df["WI_QC_REMARK"].unique()[0],
                        placeholder='Select QC remark',
                        style={'width': '250px', 'margin': '10px', 'margin-right': '10px'}
                    )
                ]),   
            ]),
            # Row 2
            html.Tr([
                html.Td([html.Label('Job Head:', style={'margin': '10px'}), ]),
                html.Td([
                    dcc.Input(id='job-head-input', type='text', placeholder='Enter job head', style={'width': '300px', 'margin': '10px', 'margin-right': '20px'})
                ]),
                html.Td([html.Label('Select Type of Equipment:', style={'margin': '10px'}), ]),
                html.Td([
                    dcc.Dropdown(
                        id='categorical-dropdown-equipment',
                        options=[{'label': col, 'value': col} for col in df["EQUIPMENT_NAME"].unique()],
                        value=df["EQUIPMENT_NAME"].unique()[0],
                        placeholder='Select equipment name',
                        style={'width': '250px', 'margin': '10px', 'margin-right': '10px'}
                    )
                ]),
            ]),
            # Row 3
            html.Tr([
                html.Td([html.Label('Job Detail:', style={'margin': '10px'}), ]),
                html.Td([
                    dcc.Input(id='job-detail-input', type='text', placeholder='Enter job detail', style={'width': '300px', 'margin': '10px', 'margin-right': '20px'})
                ]),
                html.Td([html.Label('Select Refitcode:', style={'margin': '10px'}), ]),
                html.Td([
                    dcc.Dropdown(
                        id='categorical-dropdown-refitcode',
                        options=[{'label': col, 'value': col} for col in df["STR_REFIT_CODE"].unique()],
                        value=df["STR_REFIT_CODE"].unique()[0],
                        placeholder='Select a refit code',
                        style={'width': '250px', 'margin': '10px', 'margin-right': '10px'}
                    )
                ]),
            ]),
            # Row 4
            html.Tr([
                html.Td([html.Label('Job Summary:', style={'margin': '10px'}), ]),
                html.Td([
                    dcc.Input(id='job-sum-input', type='text', placeholder='Enter job summary', style={'width': '300px', 'margin': '10px', 'margin-right': '20px'})
                ]),
                html.Td([html.Label('Select Offloaded:', style={'margin': '10px'}), ]), 
                html.Td([
                    dcc.Dropdown(
                        id='categorical-dropdown-offloaded',
                        options=[{'label': col, 'value': col} for col in df["FLG_OFFLOADED"].unique()],
                        value=df["FLG_OFFLOADED"].unique()[0],
                        placeholder='Select offloaded status',
                        style={'width': '250px', 'margin': '10px', 'margin-right': '10px'}
                    )
                ]),
            ]),

        ], style={'border': 'none','margin-left': '50px'}),
    ], className='row', style={'backgroundColor': '#F2F2F2', 'borderRadius': '15px', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id='prediction-plot', className='container'),
    ], className='container'),
], style={'backgroundColor': '#E5E5E5', 'padding': '20px'})

# Callback to update the graph based on the input and dropdown selections
@app.callback(
    [Output('prediction-plot', 'figure')],
    [Input('defect-desc-input', 'value'),
     Input('job-detail-input', 'value'),
     Input('job-sum-input', 'value'),
     Input('job-head-input', 'value'),
     Input('categorical-dropdown-shipname', 'value'),
     Input('categorical-dropdown-refitcode', 'value'),
     Input('categorical-dropdown-offloaded', 'value'),
     Input('categorical-dropdown-equipment', 'value'),
     Input('categorical-dropdown-qcremark', 'value')]
)
def update_graph(defect_desc, job_detail, job_sum, job_head, selected_shipname, selected_refitcode, selected_offloaded, selected_equipment, selected_qcremark):
    if defect_desc and job_detail and job_sum and job_head and selected_shipname and selected_refitcode and selected_offloaded and selected_equipment and selected_qcremark:
        # Create a DataFrame for the input features
        input_data = pd.DataFrame({
            'DEFECT_DESC': [defect_desc],
            'JOBDETAIL': [job_detail],
            'JOBSUM': [job_sum],
            'JOBHEAD': [job_head],
            'STR_SHIPNAME': [selected_shipname],
            'STR_REFIT_CODE': [selected_refitcode],
            'FLG_OFFLOADED': [selected_offloaded],
            'EQUIPMENT_NAME': [selected_equipment],
            'WI_QC_REMARK': [selected_qcremark]
        })

        # Make predictions for the transformed input data
        prediction = model.predict(input_data)

        # Generate a normal distribution curve for target column "EMD"
        mu, sigma = 500, 150  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)

        # Create a histogram using Plotly
        hist_data = [s]
        group_labels = ['distplot']

        fig = go.Figure(data=[go.Histogram(x=s, nbinsx=30, name='Actual Data')])
        fig.add_trace(go.Scatter(x=[prediction[0], prediction[0]], y=[0, 0.0025], mode='lines', name='Predicted Value'))
        #fig.add_trace(go.Scatter(x=[prediction[0]], y=[0.0025], mode='markers', marker=dict(size=10, color='red'), name='Predicted Point'))
        fig.add_trace(go.Scatter(x=[prediction[0], prediction[0]], y=[0, 0.0025], mode='markers', marker=dict(size=10, color='red'), name='Predicted Point'))
        fig.update_layout(
            title_text='Normal Distribution Curve for EMD with Predicted Value',
            xaxis_title_text='EMD',
            yaxis_title_text='Density',
            bargap=0.2,
            bargroupgap=0.1
        )

        return [fig]
    else:
        # If no input is provided, return an empty graph
        return [px.scatter()]

# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True)

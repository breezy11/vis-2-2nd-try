import dash
from dash import dcc, html, Input, Output, callback_context
import json
import string
import random
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import os
import base64

dataset_names = ['cars', 'cars_year', 'iris', 'seeds', 'glass', 'ecoli']

dataset_dicts = {}
datasets = {}
targets_dict = {'cars': 'origin',
                'cars_year': 'model year',
                'ecoli': 'class',
                'glass': 'Type of glass',
                'iris': 'class',
                'seeds': 'class' }

for dataset_name in dataset_names:
    datasets[dataset_name] = pd.read_csv(f'data/{dataset_name}/{dataset_name}.csv', index_col=0)
    for fs in ['FS1', 'FS2', 'FS3']:
        json_file_path = f'results/{dataset_name}/{fs}_results_{dataset_name}_cols.json'
        with open(json_file_path, 'r') as json_file:
            dataset_dict = json.load(json_file)
            dataset_dicts[f'{dataset_name}_{fs}'] = dataset_dict

tds_values_list = [item['tds'] for item in dataset_dicts['cars_FS1'].values()]
tds_values_list = sorted(list(set(tds_values_list)))

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H2(
        children='Reordering Sets of Parallel Coordinates Plots to Highlight Differences in Clusters',
        style={'textAlign': 'center'}
    ),
    html.H3(children='Select the data set and the Feature Signature', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {'label': 'Cars', 'value': 'cars'},
                    {'label': 'Cars year', 'value': 'cars_year'},
                    {'label': 'Ecoli', 'value': 'ecoli'},
                    {'label': 'Glass', 'value': 'glass'},
                    {'label': 'Iris', 'value': 'iris'},
                    {'label': 'Seeds', 'value': 'seeds'}
                ],
                value='cars',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'textAlign': 'center'}),

        html.Div([
            dcc.RadioItems(
                id='fs-radio',
                options=[
                    {'label': 'FS1', 'value': 'FS1'},
                    {'label': 'FS2', 'value': 'FS2'},
                    {'label': 'FS3', 'value': 'FS3'}
                ],
                value='FS1',
                inline=True,
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'textAlign': 'center'})
    ], style={'width': '100%', 'display': 'flex'}),
    html.H4(id='tds-header', children=[], style={'textAlign': 'center'}),
    dcc.Slider(
            id='my-slider',
            min=0,
            max=len(tds_values_list) - 1,
            step=1,
            value=0,
            marks=None
        ),
    html.Div(id='return_div', style={'textAlign': 'center'})
])

@app.callback(
    [Output('return_div', 'children'), Output('tds-header', 'children'), Output('my-slider', 'max'), Output('my-slider', 'value')],
    [Input('my-dropdown', 'value'), Input('my-slider', 'value'), Input('fs-radio', 'value')]
)
def update_output(dropdown, slider_value, fs_chosen):

    triggered_id = callback_context.triggered_id

    if triggered_id is None or triggered_id == 'my-dropdown' or triggered_id == 'fs-radio':
        slider_value = 0

    # Delete all .png files in the /plots folder
    for file_name in os.listdir('plots'):
        if file_name.endswith('.png'):
            os.remove(os.path.join('plots', file_name))

    df = datasets[dropdown]
    target = targets_dict[dropdown]
    tds_list = dataset_dicts[f'{dropdown}_{fs_chosen}']

    tds_values_list = [item['tds'] for item in tds_list.values()]
    tds_values_list = sorted(list(set(tds_values_list)))

    tds_score = tds_values_list[slider_value]

    for key, value in tds_list.items():
        if value.get('tds') == tds_score:
            ordering = value.get('columns_order')
            break

    ordering.append(target)

    unique_columns = []
    for column in ordering:
        if column not in unique_columns:
            unique_columns.append(column)

    ordering = unique_columns

    # List to store Graph components
    pictures = []

    # Loop through unique classes in the DataFrame
    for _class in df[target].unique().tolist():
        df_subset = df[df[target] == _class]
        df_subset = df_subset[ordering]

        fig = plt.figure(figsize=(5, 7))
        parallel_coordinates(df_subset, class_column=target)
        plt.xticks(rotation=45)
        random_string = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        file_name = f'plots/{random_string}.png'

        pictures.append(file_name)

        plt.savefig(file_name)
        plt.close()

    # Create Div components for each graph
    graph_divs = []

    # Loop through images and organize them in rows of three
    for i in range(0, len(pictures), 3):
        row_images = pictures[i:i + 3]
        row_div = html.Div([
            html.Div(children=[
                html.H4(f'Cluster {target} = {_class}'),
                html.Img(src=f"data:image/png;base64,{base64.b64encode(open(pict, 'rb').read()).decode()}")
            ], style={'textAlign': 'center'})
            for _class, pict in zip(df[target].unique().tolist(), row_images)
        ], style={'display': 'flex', 'justifyContent': 'space-around'})
        graph_divs.append(row_div)

    div_return = html.Div(children=graph_divs)

    new_slider_max = len(tds_values_list) - 1

    return div_return, f'Ordering - {ordering}, Tds-score - {tds_score}', new_slider_max, slider_value

if __name__ == '__main__':
    app.run_server(debug=True)

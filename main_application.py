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
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

with open('results/cars/FS1_results_cars_cols.json', 'r') as json_file:
    cars_dict = json.load(json_file)

with open('results/cars_year/FS1_results_cars_year_cols.json', 'r') as json_file:
    cars_year_dict = json.load(json_file)

with open('results/ecoli/FS2_results_ecoli_cols.json', 'r') as json_file:
    ecoli_dict = json.load(json_file)

with open('results/glass/FS2_results_glass_cols.json', 'r') as json_file:
    glass_dict = json.load(json_file)

with open('results/iris/FS3_results_iris_cols.json', 'r') as json_file:
    iris_dict = json.load(json_file)

with open('results/seeds/FS3_results_seeds_cols.json', 'r') as json_file:
    seeds_dict = json.load(json_file)

cars = pd.read_csv("data/cars/cars.csv", index_col=0)
cars_target_class = 'origin'

for column in cars.columns:
    if column != cars_target_class:
        cars[column] = scaler.fit_transform(cars[column].values.reshape(-1, 1))

cars_year = pd.read_csv("data/cars_year/cars_year.csv", index_col=0)
cars_year_target_class = 'model year'

for column in cars_year.columns:
    if column != cars_year_target_class:
        cars_year[column] = scaler.fit_transform(cars_year[column].values.reshape(-1, 1))

ecoli = pd.read_csv("data/ecoli/ecoli.csv", index_col=0)
ecoli_target_class = 'class'

ecoli.drop('Sequence Name', axis=1, inplace=True)

for column in ecoli.columns:
    if column != ecoli_target_class:
        ecoli[column] = scaler.fit_transform(ecoli[column].values.reshape(-1, 1))

glass = pd.read_csv("data/glass/glass.csv", index_col=0)
glass_target_class = 'Type of glass'

for column in glass.columns:
    if column != glass_target_class:
        glass[column] = scaler.fit_transform(glass[column].values.reshape(-1, 1))

iris = pd.read_csv("data/iris/iris.csv", index_col=0)
iris_target_class = 'class'

for column in iris.columns:
    if column != iris_target_class:
        iris[column] = scaler.fit_transform(iris[column].values.reshape(-1, 1))

seeds = pd.read_csv("data/seeds/seeds.csv", index_col=0)
seeds_target_class = 'class'

for column in seeds.columns:
    if column != seeds_target_class:
        seeds[column] = scaler.fit_transform(seeds[column].values.reshape(-1, 1))

tds_values_list = [item['tds'] for item in seeds_dict.values()]
tds_values_list = sorted(list(set(tds_values_list)))

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H2(
        children='Reordering Sets of Parallel Coordinates Plots to Highlight Differences in Clusters',
        style={'textAlign': 'center'}
    ),
    html.H4(children='Select the data set', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Cars', 'value': 'Cars'},
            {'label': 'Cars year', 'value': 'Cars year'},
            {'label': 'Ecoli', 'value': 'Ecoli'},
            {'label': 'Glass', 'value': 'Glass'},
            {'label': 'Iris', 'value': 'Iris'},
            {'label': 'Seeds', 'value': 'Seeds'}
        ],
        value='Cars'
    ),
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
    [Input('my-dropdown', 'value'), Input('my-slider', 'value')]
)
def update_output(dropdown, slider_value):

    triggered_id = callback_context.triggered_id

    if triggered_id is None or triggered_id == 'my-dropdown':
        slider_value = 0

    # Delete all .png files in the /plots folder
    for file_name in os.listdir('plots'):
        if file_name.endswith('.png'):
            os.remove(os.path.join('plots', file_name))

    if dropdown == 'Cars':
        df = cars
        target = cars_target_class
        tds_list = cars_dict
    elif dropdown == 'Cars year':
        df = cars_year
        target = cars_year_target_class
        tds_list = cars_year_dict
    elif dropdown == 'Ecoli':
        df = ecoli
        target = ecoli_target_class
        tds_list = ecoli_dict
    elif dropdown == 'Glass':
        df = glass
        target = glass_target_class
        tds_list = glass_dict
    elif dropdown == 'Iris':
        df = iris
        target = iris_target_class
        tds_list = iris_dict
    elif dropdown == 'Seeds':
        df = seeds
        target = seeds_target_class
        tds_list = seeds_dict

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

        fig = plt.figure(figsize=(5, 3))
        parallel_coordinates(df_subset, class_column=target)
        plt.xticks(rotation=90)
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

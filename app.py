import dash
# import datetime
import json
import pandas as pd
import dash_bootstrap_components as dbc
import dash_html_components as html
from layout import app_layout
from dash.dependencies import Input, Output, State
import dash_table
from my_functions import parse_file, split_train_test, get_color_scale, lda, make_figure, executeClassifictaionSklearn,\
    adapt_scikit_learn_colormap, check_elements
from c45_tree import createTree, setIsLeafValue, splitNodeBy, deleteNodes, \
    getData, isTreeClassifier, executeClassification, rgba_to_rgb
from sklearn.metrics import classification_report
import pydotplus
# import io


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = app_layout

# ==== COMMON FUNCTIONS ====


def train_test_dict_common(selectedFeatures, trainValue, jsonified_data, targetFeature, seed,
                           tree_jsonified, selectedNode_value):
    """
    Splits the dataset into training and testing sets based on selected features and a target feature.

    This function takes in selected features, a target feature, and optional parameters to work with a specific node in a decision tree. It retrieves the full dataset from a JSON format, potentially narrows it down to the data related to a specific node, and then splits it into training and testing sets. The resulting datasets and a color scale for visualization are returned as a JSON-serialized dictionary.

    Args:
        selectedFeatures (list): List of features to be used for training.
        trainValue (float): The proportion of the dataset to include in the train split.
        jsonified_data (str): JSON string representation of the complete dataset.
        targetFeature (str): The name of the target feature for classification.
        seed (int): Random seed for reproducibility.
        tree_jsonified (str): JSON string representation of the decision tree structure.
        selectedNode_value (str): Identifier for a specific node in the decision tree.

    Returns:
        str: A JSON string containing the training and testing sets, along with a color scale.
    """

    if jsonified_data is not None and targetFeature is not None:
        df_full = pd.read_json(jsonified_data, orient='split')
        # Only if selected node is not Root then we must take a subset of the dataframe
        if selectedNode_value is not None:
            if tree_jsonified is not None:
                tree = json.loads(tree_jsonified)
                for node in tree['nodes']:
                    if node['id'] == selectedNode_value:
                        if not node['isRoot']:
                            # Extract the index for the observations on the selected tree node
                            # tree_df_jsonified = tree['data']['df']
                            # tree_df = pd.read_json(tree_df_jsonified, orient='split')
                            node_df, _, _ = getData(tree, selectedNode_value)
                            node_df_index = node_df.index.tolist()
                            df = df_full.loc[node_df_index, :]
                            full_train_check = True
        else:
            df = df_full
            full_train_check = False

        X_train, X_test, y_train, y_test = split_train_test(df, targetFeature, trainValue, seed, full_train_check)
        color_scale = get_color_scale(len(tree['data']['classes']))
        train_test_dict = {'X_train': X_train.to_json(orient='split'),
                           'X_test': X_test.to_json(orient='split'),
                           'y_train': y_train.to_json(orient='split'),
                           'y_test': y_test.to_json(orient='split'),
                           'color_scale': color_scale}

        return json.dumps(train_test_dict)


def execute_lda_common(train_test_json, targetFeature, features, seed, complete_json, num_classes, jittering):
    """
    Executes Linear Discriminant Analysis (LDA) on the provided training and testing datasets.

    This function performs LDA on the datasets parsed from the input JSON strings, and prepares the results for visualization or further analysis. It retrieves the training and testing datasets along with the original dataset, processes them for LDA, and returns a JSON-serialized dictionary containing all relevant information.

    Args:
        train_test_json (str): JSON string representation of the training and testing datasets.
        targetFeature (str): The name of the target feature for classification.
        features (list): List of features to be used for LDA.
        seed (int): Random seed for reproducibility.
        complete_json (str): JSON string representation of the complete dataset.
        num_classes (int): Number of unique classes in the target feature.
        jittering (bool): Indicates whether to apply jittering for visualization.

    Returns:
        str: A JSON string containing the results of the LDA, including transformed datasets and other related data.
    """

    if train_test_json is not None:
        train_test = json.loads(train_test_json)
        y_train = pd.read_json(train_test['y_train'], orient='split', typ='series')
        y_test = pd.read_json(train_test['y_test'], orient='split', typ='series')
        X_train = pd.read_json(train_test['X_train'], orient='split')
        X_test = pd.read_json(train_test['X_test'], orient='split')

        X_train_index = X_train.index.tolist()
        X_test_index = X_test.index.tolist()

        raw_dataset = pd.read_json(complete_json, orient='split')
        X_train_original = raw_dataset.loc[X_train_index, :]
        X_test_original = raw_dataset.loc[X_test_index, :]

        features.append('index')

        color_scale = train_test['color_scale']
        lda_dict = {}
        lda_dict['color_scale'] = color_scale

        Xr_train, Xr_test, axis, num_classes = lda(X_train, X_test, y_train, y_test, features,
                                                   seed, num_classes, jittering)

        lda_dict['lda'] = {
            'X_train': pd.DataFrame(X_train).to_json(orient='split'),
            'X_test': pd.DataFrame(X_test).to_json(orient='split'),
            'X_train_not_scaled': pd.DataFrame(X_train_original).to_json(orient='split'),
            'X_test_not_scaled': pd.DataFrame(X_test_original).to_json(orient='split'),
            'Xr_train': pd.DataFrame(Xr_train).to_json(orient='split'),
            'Xr_test': pd.DataFrame(Xr_test).to_json(orient='split'),
            'y_train': y_train.to_json(orient='split'),
            'y_test': y_test.to_json(orient='split'),
            'axis': pd.DataFrame(axis).to_json(orient='split'),
            'features': features,
            'target_feature': targetFeature,
            'num_classes': num_classes
        }

        return json.dumps(lda_dict)

# ==== CALLBACKS FOR NAVBAR ====


PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"),
     Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):

    if n1 or n2:
        return not is_open
    return is_open

# ==== CALLBACKS FOR UPLOAD FILE ====


# Save the dataframe in hidden Div
@app.callback(Output('dataframe', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('field-sep', 'value')])
def generate_df(contents, filename, separator):

    if contents is not None:
        full_df = parse_file(contents, filename, separator)
        if full_df is not None:
            full_df = full_df.reset_index(level=0, drop=False)
            return full_df.to_json(orient='split')


# Fix seed when dataset is loaded
@app.callback(Output('seed', 'children'),
              [Input('upload-data', 'filename')])
def create_seed(filename):

    if filename is not None:
        # now = datetime.datetime.now()
        # seed = now.hour*10000+now.minute*100+now.second
        seed = 42
        return seed


# Show uploaded file name
@app.callback(Output('file-name', 'children'),
              [Input('upload-data', 'filename')])
def update_filename(name):

    if name is not None:
        return name

# ==== CALLBACKS FOR TARGET FEATURE SELECTION ====


# Activate the target dropdown menu
@app.callback(Output('target-feature', 'disabled'),
              [Input('dataframe', 'children')])
def activate_targetFeature_dropdown(jsonified_data):

    if jsonified_data is not None:
        return False
    else:
        return True


# Update target features
@app.callback(Output('target-feature', 'options'),
              [Input('dataframe', 'children')])
def update_targetFeature_options(jsonified_data):

    if jsonified_data is not None:
        data = json.loads(jsonified_data)['columns']
        data.remove('index')
        diccionary_options = [{'label': i, 'value': i} for i in data]
        return diccionary_options
    else:
        return []

# ==== CALLBACKS FOR SELECTED FEATURES ====


# Activate selected features dropdown
@app.callback(Output('selected-features', 'disabled'),
              [Input('target-feature', 'value')])
def activate_selectedFeatures_dropdown(target_feature):
    if target_feature is not None:
        return False
    else:
        return True


# Load options for selected features dropdown
@app.callback(Output('selected-features', 'options'),
              [Input('target-feature', 'value'),
              Input('target-feature', 'options')])
def generate_selectedFeatures_options(target_feature, options):
    if options is not None and target_feature is not None:
        return [i for i in options if (i['value'] != target_feature)]
    else:
        return []


# Set all options to value for selected features dropdown
@app.callback(Output('selected-features', 'value'),
              [Input('selected-features', 'options')])
def update_selectedFeatures_options(options):
    if options is not None:
        return [i['value'] for i in options]
    else:
        return []


# Generate df_to_use and save it in hidden Div
@app.callback(Output('df_to_use', 'children'),
              [Input('selected-features', 'value')],
              [State('dataframe', 'children'),
               State('target-feature', 'value')])
def generate_df(selectedFeatures, jsonified_data, targetFeature):

    if len(selectedFeatures) != 0:
        df = pd.read_json(jsonified_data, orient='split')
        new_cols = [i for i in selectedFeatures]
        new_cols += [targetFeature, 'index']

        return df[new_cols].to_json(orient='split')

# ==== CALLBACKS FOR TRAIN TEST DIVISION CONTROLS ====


# Activate knob
@app.callback(Output('train-knob', 'disabled'),
              [Input('target-feature', 'value')])
def activate_train_knob(target_feature):
    if target_feature is not None:
        return False
    else:
        return True


# Activate knob
@app.callback(Output('train-knob', 'value'),
              [Input('target-feature', 'value')])
def activate_train_knob(target_feature):
    if target_feature is None:
        return 0


# Move graduatebar according to knob
@app.callback(Output('train_bar', 'value'),
              [Input('train-knob', 'value')])
def move_train_bar(knob_value):
    return knob_value


# ==== GENERATE TRAIN/TEST DATAFRAME ====

# Generate train_test_df and save it in hidden Div
@app.callback(Output('train_test_df', 'children'),
              [Input('selected-features', 'value'),
               Input('train_bar', 'value')],
              [State('df_to_use', 'children'),
               State('target-feature', 'value'),
               State('seed', 'children')])
def train_test_dict(selectedFeatures, trainValue, jsonified_data, targetFeature, seed):

    if jsonified_data is not None and targetFeature is not None and (trainValue != 0 and trainValue is not None):
        df_full = pd.read_json(jsonified_data, orient='split')

        X_train, X_test, y_train, y_test = split_train_test(df_full, targetFeature, trainValue, seed)
        color_scale = get_color_scale(len(y_train.unique()))
        train_test_dict = {'X_train': X_train.to_json(orient='split'),
                           'X_test': X_test.to_json(orient='split'),
                           'y_train': y_train.to_json(orient='split'),
                           'y_test': y_test.to_json(orient='split'),
                           'color_scale': color_scale}

        return json.dumps(train_test_dict)


# ==== EXECUTE LDA ALGORITHM ====

@app.callback(Output('lda_dict', 'children'),
              [Input('train_test_df', 'children')],
              [State('target-feature', 'value'),
               State('selected-features', 'value'),
               State('seed', 'children'),
               State('df_to_use', 'children')])
def execute_lda(train_test_json, targetFeature, features, seed, complete_json):

    result = execute_lda_common(train_test_json, targetFeature, features, seed, complete_json, None, jittering=True)

    if result is not None:
        return result


# ==== PLOT LDA DATA ====

@app.callback(Output('lda-plot', 'figure'),
              [Input('lda_dict', 'children'),
               Input('tree_cyto', 'tapNodeData'),
               Input('jittering_checklist', 'value')],
              [State('tree-dict', 'children'),
               State('selected-features', 'value'),
               State('train_bar', 'value'),
               State('df_to_use', 'children'),
               State('target-feature', 'value'),
               State('seed', 'children')])
def make_lda_plot(lda_json, selectedNode_data, jittering, tree_jsonified, selectedFeatures, trainValue, jsonified_data,
                  targetFeature, seed):

    if jittering is None or not jittering:
        jittering = False
    else:
        jittering = True

    if lda_json is not None:
        if tree_jsonified is not None:
            tree = json.loads(tree_jsonified)
            if selectedNode_data is not None:
                if len(selectedNode_data['id']) != 0:
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_data['id']:
                            if not node['isRoot']:
                                # Need to check the number of classes of the node
                                # If one single class, no plot is generated
                                node_df, _, _ = getData(tree, node['id'])
                                num_classes = len(node_df[targetFeature].unique())
                                if num_classes == 1:
                                    return {}
                                else:
                                    # New lda-dict needs to be build
                                    train_test_dict_jsonified = train_test_dict_common(selectedFeatures, trainValue, jsonified_data,
                                                                                       targetFeature, seed, tree_jsonified,
                                                                                       selectedNode_data['id'])
                                    new_lda_dict_json = execute_lda_common(train_test_dict_jsonified, targetFeature, selectedFeatures,
                                                                           seed, jsonified_data, num_classes, jittering)
                                    # We send the original classes list to get original rgba codes
                                    classes = tree['data']['classes']
                                    figure = make_figure(new_lda_dict_json, classes, jittering)
                                    return figure

        figure = make_figure(lda_json, None, True)
        return figure
    else:
        return {}


@app.callback(Output('lda-plot', 'style'),
              [Input('lda-plot', 'figure')])
def show_visualizations(figure):
    if figure is not None:
        if len(figure) != 0:
            return dict()

    return dict(visibility='hidden')


# ==== INCLUDE LDA TABLE FOR 1-D CASE ====

@app.callback(Output('table', 'children'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('jittering_checklist', 'value')],
              [State('tree-dict', 'children'),
               State('selected-features', 'value'),
               State('train_bar', 'value'),
               State('df_to_use', 'children'),
               State('target-feature', 'value'),
               State('seed', 'children')])
def make_table(selectedNode_data, jittering, tree_jsonified, selectedFeatures, trainValue, jsonified_data,
                  targetFeature, seed):

    if jittering is None or not jittering:
        jittering = False
    else:
        jittering = True
        return []

    if selectedNode_data is not None:
        if len(selectedNode_data['id']) != 0:
            tree = json.loads(tree_jsonified)
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    if not node['isRoot']:
                        node_df, _, _ = getData(tree, node['id'])
                        num_classes = len(node_df[targetFeature].unique())
                        if num_classes == 2 and ~jittering:
                            # New lda-dict needs to be build
                            train_test_dict_jsonified = train_test_dict_common(selectedFeatures, trainValue, jsonified_data,
                                                                               targetFeature, seed, tree_jsonified,
                                                                               selectedNode_data['id'])
                            new_lda_dict_json = execute_lda_common(train_test_dict_jsonified, targetFeature, selectedFeatures,
                                                                   seed, jsonified_data, num_classes, jittering)
                            data_dict = json.loads(new_lda_dict_json)
                            lda_dict=data_dict['lda']
                            features = lda_dict['features']
                            if 'index' in features:
                                features.remove('index')
                                axis = pd.read_json(lda_dict['axis'], orient='split')
                                values = [str(round(i, 2)) for i in axis.values[0].tolist()]
                                if len(values) == len(features):
                                    cols = [{
                                            'name': col,
                                            'id': col
                                    } for col in features]
                                    data = dict()
                                    for i in range(len(values)):
                                        data[features[i]] = values[i]
                                    features_table = dash_table.DataTable(
                                        columns=cols,
                                        data=[data],
                                        style_table={
                                            'overflowX': 'scroll',
                                        },
                                        style_header={'fontWeight': 'bold'}
                                    )
                                    return [features_table]
                                    # return html.Table(
                                    #             # Header
                                    #             [html.Tr([html.Th(col) for col in features]) ] +
                                    #             # Body
                                    #             [html.Tr([html.Td(value) for value in values])],
                                    #             className='table'
                                    #         )

# ==== CALLBACKS FOR DECISION TREE CONTROLS ====


# Activate the chosenAttr-options
@app.callback(Output('chosenAttr-options', 'disabled'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def activate_chosenAttr_dropdown(selectedNode_data, tree_jsonified, selectedNodeData_list):

    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            tree = json.loads(tree_jsonified)
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    if node['hasChildren'] or node['isLeaf']:
                        return True
                    else:
                        return False
        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    if len(selectedNodeData_list) == 1:
                        selectedNode_value = selectedNodeData_list[0]['id']
                    else:
                        selectedNode_value = selectedNodeData_list[-1]['id']
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_value:
                            if node['hasChildren'] or node['isLeaf']:
                                return True
                            else:
                                return False

    return True


# Set the chosenAttr-options options
@app.callback(Output('chosenAttr-options', 'options'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def options_chosenAttr_dropdown(selectedNode_data, tree_jsonified, selectedNodeData_list):
    # Only if there is a node selected we must set the chosen Attr options
    #
    # This callback must be triggered by:
    #  1.- Direct user interaction: The user selects a node
    #  2.- Indirect user interaction: An action executed by the user leads to changes
    #      that affect this callback
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            tree = json.loads(tree_jsonified)
            selectedFeatures = tree['data']['selectedFeatures']
            diccionary_options = [{'label': feature, 'value': feature} for feature in selectedFeatures
                                  if (feature != 'index' and feature != 'cluster_class')]
            return diccionary_options
        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    selectedFeatures = tree['data']['selectedFeatures']
                    diccionary_options = [{'label': feature, 'value': feature} for feature in selectedFeatures
                                          if (feature != 'index' and feature != 'cluster_class')]
                    return diccionary_options

    return []


# Callback to set the value of chosenAttr when node has been already splited
@app.callback(Output('chosenAttr-options', 'value'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def value_chosenAttr_dropdown(selectedNode_data, tree_jsonified, selectedNodeData_list):
    # Only if there is a node selected that has been already splited
    # we must set the chosen Attr value
    #
    # This callback must be triggered by:
    #  1.- Direct user interaction: The user selects a node
    #  2.- Indirect user interaction: An action executed by the user leads to changes
    #      that affect this callback
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    if node['chosenAttr'] is not None:
                        return node['chosenAttr']
        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    if len(selectedNodeData_list) == 1:
                        selectedNode_value = selectedNodeData_list[0]['id']
                    else:
                        selectedNode_value = selectedNodeData_list[-1]['id']
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_value:
                            if node['chosenAttr'] is not None:
                                return node['chosenAttr']

    return None


# Callback to set the button CREATE to disabled
@app.callback(Output('createNode-button', 'disabled'),
              [Input('chosenAttr-options', 'value'),
               Input('tree_cyto', 'selectedNodeData')],
              [State('chosenAttr-options', 'disabled'),
               State('tree-dict', 'children')])
def disable_Create_button(chosenAttr, selectedNodeData_list, chosenAttr_disabled, tree_jsonified):
    # The conditions to enable the CREATE node button are:
    #  1.- There is a node selected (always true when there is a chosenAttr value set)
    #  2.- The node has NOT children YET
    #  3.- There is a chosen feature selected

    if not chosenAttr_disabled:
        if chosenAttr is not None:
            tree = json.loads(tree_jsonified)
            # Check which is the selected Node
            if len(selectedNodeData_list) !=0:
                if len(selectedNodeData_list) == 1:
                    selectedNode_value = selectedNodeData_list[0]['id']
                else:
                    selectedNode_value = selectedNodeData_list[-1]['id']
                for node in tree['nodes']:
                    if node['id'] == selectedNode_value:
                        if node['hasChildren']:
                            return True
                        else:
                            return False

    return True


# Callback to set the button DELETE to disabled
@app.callback(Output('deleteNode-button', 'disabled'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def disable_Delete_button(selectedNode_data, tree_jsonified, selectedNodeData_list):
    # The conditions to enable the DELETE node button are:
    #  1.- There is a node selected
    #  2.- The node has children ALREADY
    #
    # This callback must be triggered by:
    #  1.- Direct user interaction: The user selects a node
    #  2.- Indirect user interaction: An action executed by the user leads to changes
    #      that affect this callback
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            tree = json.loads(tree_jsonified)
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    if node['hasChildren']:
                        return False
                    else:
                        return True
        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    if len(selectedNodeData_list) == 1:
                        selectedNode_value = selectedNodeData_list[0]['id']
                    else:
                        selectedNode_value = selectedNodeData_list[-1]['id']
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_value:
                            if node['hasChildren']:
                                return False
                            else:
                                return True

    return True


# Callback to set the button FORCE LEAF to disabled
@app.callback(Output('forceLeaf-button', 'disabled'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def disable_forceLeaf_button(selectedNode_data, tree_jsonified, selectedNodeData_list):
    # The conditions to enable the FORCE LEAF node button are:
    #  1.- There is a node selected
    #  2.- The node has NOT children ALREADY
    #  3.- The node is not LEAF yet
    #
    # This callback must be triggered by:
    #  1.- Direct user interaction: The user selects a node
    #  2.- Indirect user interaction: An action executed by the user leads to changes
    #      that affect this callback
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    if not node['hasChildren'] and not node['isLeaf']:
                        return False
                    else:
                        return True

        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    if len(selectedNodeData_list) == 1:
                        selectedNode_value = selectedNodeData_list[0]['id']
                    else:
                        selectedNode_value = selectedNodeData_list[-1]['id']
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_value:
                            if not node['hasChildren'] and not node['isLeaf']:
                                return False
                            else:
                                return True

    return True


# Callback to set the button RESET LEAF to disabled
@app.callback(Output('resetLeaf-button', 'disabled'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def disable_resetLeaf_button(selectedNode_data, tree_jsonified, selectedNodeData_list):
    # The conditions to enable the RESET LEAF node button are:
    #  1.- There is a node selected
    #  2.- The node is LEAF
    #  3.- The node has not any class purity value equqal to 1
    #
    # This callback must be triggered by:
    #  1.- Direct user interaction: The user selects a node
    #  2.- Indirect user interaction: An action executed by the user leads to changes
    #      that affect this callback
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    if node['isLeaf']:
                        for item in node['classPurity']:
                            if item == 1:
                                return True
                        return False
                    else:
                        return True
        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    if len(selectedNodeData_list) == 1:
                        selectedNode_value = selectedNodeData_list[0]['id']
                    else:
                        selectedNode_value = selectedNodeData_list[-1]['id']
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_value:
                            if node['isLeaf']:
                                for item in node['classPurity']:
                                    if item == 1:
                                        return True
                                return False
                            else:
                                return True

    return True


# Callbacks to update the Text DIVs containing Node Information
@app.callback(Output('text-area', 'children'),
              [Input('tree_cyto', 'tapNodeData'),
               Input('tree-dict', 'children')],
              [State('tree_cyto', 'selectedNodeData')])
def updateNodeInfo_TextArea(selectedNode_data, tree_jsonified, selectedNodeData_list):
    # This callback must be triggered by:
    #  1.- Direct user interaction: The user selects a node
    #  2.- Indirect user interaction: An action executed by the user leads to changes
    #      that affect this callback
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # Direct user interaction
        if selectedNode_data is not None:
            for node in tree['nodes']:
                if node['id'] == selectedNode_data['id']:
                    node_id = node['id']
                    is_root = str(node['isRoot'])
                    if node['class'] is not None:
                        nodeClass = str(int(node['class']))
                    else:
                        nodeClass = str(node['class'])
                    purity = str([round(x, 2) for x in node['classPurity']])
                    entropy = str(round(node['entropy'], 2))
                    is_leaf = str(node['isLeaf'])
                    if node['hasChildren']:
                        childs = str(node['leftChild']) + " / " + str(node['rightChild'])
                        chosenAttr = str(node['chosenAttr'])
                        threshold = str(round(node['threshold'],1))
                    else:
                        childs = ''
                        chosenAttr = ''
                        threshold = ''
                    children = [
                        html.Div(children=[
                            html.Div("Node Id: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[node_id], style={'display': 'inline'})
                            ]),
                        html.Div(children=[
                            html.Div("Is Root: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[is_root], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Class Purity: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[purity], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Entropy: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[entropy], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Is Leaf: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[is_leaf], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Class: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[nodeClass], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Left/Right Childs: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[childs], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Partitioning Feature: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[chosenAttr], style={'display': 'inline'})
                        ]),
                        html.Div(children=[
                            html.Div("Threshold Value: ", style={'font-weight': 'bold', 'display': 'inline'}),
                            html.Div(children=[threshold], style={'display': 'inline'})
                        ])

                    ]
                    return children
        # Indirect user interaction
        else:
            # Check which is the selected Node
            if selectedNodeData_list is not None:
                if len(selectedNodeData_list) != 0:
                    if len(selectedNodeData_list) == 1:
                        selectedNode_value = selectedNodeData_list[0]['id']
                    else:
                        selectedNode_value = selectedNodeData_list[-1]['id']
                    for node in tree['nodes']:
                        if node['id'] == selectedNode_value:
                            node_id = node['id']
                            is_root = str(node['isRoot'])
                            if node['class'] is not None:
                                nodeClass = str(int(node['class']))
                            else:
                                nodeClass = str(node['class'])
                            purity = str([round(x, 2) for x in node['classPurity']])
                            entropy = str(round(node['entropy'], 2))
                            is_leaf = str(node['isLeaf'])
                            if node['hasChildren']:
                                childs = str(node['leftChild']) + " / " + str(node['rightChild'])
                                chosenAttr = str(node['chosenAttr'])
                                threshold = str(round(node['threshold'], 1))
                            else:
                                childs = ''
                                chosenAttr = ''
                                threshold = ''
                            children = [
                                html.Div(children=[
                                    html.Div("Node Id: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[node_id], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Is Root: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[is_root], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Class Purity: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[purity], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Entropy: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[entropy], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Is Leaf: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[is_leaf], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Class: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[nodeClass], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Left/Right Childs: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[childs], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Partitioning Feature: ",
                                             style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[chosenAttr], style={'display': 'inline'})
                                ]),
                                html.Div(children=[
                                    html.Div("Threshold Value: ", style={'font-weight': 'bold', 'display': 'inline'}),
                                    html.Div(children=[threshold], style={'display': 'inline'})
                                ])

                            ]
                            return children

    children = [
        html.Div(children=[
            html.Div("Node Id: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Is Root: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Class Purity: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Is Leaf: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Class: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Left/Right Childs: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Partitioning Feature: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ]),
        html.Div(children=[
            html.Div("Threshold Value: ", style={'font-weight': 'bold', 'display': 'inline'}),
            html.Div(children=[], style={'display': 'inline'})
        ])

    ]
    return children


# Main Callback: Generate and Modify Tree dictionary
@app.callback(Output('tree-dict', 'children'),
              [Input('lda_dict', 'children'),
               Input('createNode-button', 'n_clicks'),
               Input('deleteNode-button', 'n_clicks'),
               Input('forceLeaf-button', 'n_clicks'),
               Input('resetLeaf-button', 'n_clicks')],
              [State('tree_cyto', 'selectedNodeData'),
               State('chosenAttr-options', 'value'),
               State('tree-dict', 'children')])
def generate_modify_tree(jsonified_data, create_btn, delete_btn, forceLeaf_btn,
                         resetLeaf_btn, selectedNodeData_list, chosen_Attr, old_tree_jsonified):

    # Delete the tree when the algorithm dictionary does not exist or has been deleted
    if jsonified_data is None:
        return None

    # Check first if any of the buttons have been clicked
    if old_tree_jsonified is not None:
        # Check which is the selected Node
        if selectedNodeData_list is not None:
            if len(selectedNodeData_list) != 0:
                if len(selectedNodeData_list) == 1:
                    selectedNode_value = selectedNodeData_list[0]['id']
                else:
                    selectedNode_value = selectedNodeData_list[-1]['id']

        # Read the current tree_dict
        old_tree = json.loads(old_tree_jsonified)
        old_tree_create_btn = old_tree['clicks']['create']
        old_tree_delete_btn = old_tree['clicks']['delete']
        old_tree_forceLeaf_btn = old_tree['clicks']['forceLeaf']
        old_tree_resetLeaf_btn = old_tree['clicks']['resetLeaf']
         # Compare with current clicks
        if create_btn > old_tree_create_btn:
            # Create button has been pressed
            # We want to split selected_Node by the feature chosen_Attr
            # and generate a new tree
            for node in old_tree['nodes']:
                if node['id'] == selectedNode_value:
                    new_tree, result, result_info = splitNodeBy(old_tree, selectedNode_value, chosen_Attr, create_btn,
                                                                delete_btn, forceLeaf_btn, resetLeaf_btn)
                    if result:
                        return json.dumps(new_tree)
                    else:
                        print("ERROR Creating new Nodes!: " + result_info)
                        exit()
        elif delete_btn > old_tree_delete_btn:
            # Delete button has been pressed
            # We want to delete all the nodes below the selected_Node
            # and update the tree accordingly
            for node in old_tree['nodes']:
                if node['id'] == selectedNode_value:
                    new_tree, result, result_info = deleteNodes(old_tree, selectedNode_value, create_btn, delete_btn,
                                                                forceLeaf_btn, resetLeaf_btn)
                    if result:
                        return json.dumps(new_tree)
                    else:
                        print("ERROR Deleting Nodes!: " + result_info)
                        exit()
        elif forceLeaf_btn > old_tree_forceLeaf_btn:
            # Force Leaf button has been pressed
            # We want to make the current Node a Leaf Node and
            # update the tree accordingly
            for node in old_tree['nodes']:
                if node['id'] == selectedNode_value:
                    new_tree, result, result_info = setIsLeafValue(old_tree, selectedNode_value, True,
                                                                create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn)
                    if result:
                        return json.dumps(new_tree)
                    else:
                        print("ERROR Forcing Node to be Leaf!: " + result_info)
                        exit()
        elif resetLeaf_btn > old_tree_resetLeaf_btn:
            # Reset Leaf button has been pressed
            # We want to unmark the current Node as Leaf Node and
            # update the tree accordingly
            for node in old_tree['nodes']:
                if node['id'] == selectedNode_value:
                    new_tree, result, result_info = setIsLeafValue(old_tree, selectedNode_value, False,
                                                                   create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn)
                    if result:
                        return json.dumps(new_tree)
                    else:
                        print("ERROR Forcing Node to reset Leaf!: " + result_info)
                        exit()

    # None of the buttons have been pressed. Continue.
    # Generate tree and root node when nothing is still stored
    if jsonified_data is not None and old_tree_jsonified is None:
        total_color_scale = json.loads(jsonified_data)['color_scale']
        lda_dict = json.loads(jsonified_data)['lda']
        X_train_not_scaled_json = lda_dict['X_train_not_scaled']
        X_train_not_scaled = pd.read_json(X_train_not_scaled_json, orient='split')
        selectedFeatures = lda_dict['features']
        targetFeature = lda_dict['target_feature']

        tree_dict = createTree(X_train_not_scaled, targetFeature, selectedFeatures,
                               create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn, total_color_scale)
        return json.dumps(tree_dict)

    # Compare main parameters when between algorithm dictionary and existing tree
    # If they match, do nothing, if they differ, tree must be deleted and new one created
    # with new parameters, only having Root node
    if jsonified_data is not None and old_tree_jsonified is not None:
        # Read run-algorithms-dict
        lda_dict = json.loads(jsonified_data)['lda']
        total_color_scale = json.loads(jsonified_data)['color_scale']

        X_train_not_scaled_json = lda_dict['X_train_not_scaled']
        X_train_not_scaled = pd.read_json(X_train_not_scaled_json, orient='split')
        algo_trainLength = len(X_train_not_scaled)
        algo_selectedFeatures = lda_dict['features']
        algo_targetFeature = lda_dict['target_feature']


        # Read the current tree_dict
        old_tree = json.loads(old_tree_jsonified)
        old_tree_trainLenght = old_tree['data']['train_length']
        old_tree_selectedFeatures = old_tree['data']['selectedFeatures']
        old_tree_targetFeature = old_tree['data']['targetFeature']
        # Compare both: If they differ, create new tree from scratch with new data
        if old_tree_trainLenght != algo_trainLength or old_tree_selectedFeatures != algo_selectedFeatures \
            or old_tree_targetFeature != algo_targetFeature:
            new_tree_dict = createTree(X_train_not_scaled, algo_targetFeature, algo_selectedFeatures,
                                       create_btn, delete_btn, forceLeaf_btn, resetLeaf_btn, total_color_scale)
            return json.dumps(new_tree_dict)
        else:
            return old_tree_jsonified


# ==== CALLBACKS FOR CYTOSCAPE OBJECT PLOT ====

def add_two_nodes(tree_nodes, cyto_nodes_id):
    """
    This function takes a list of nodes from the tree and the ids of all the nodes
    of a cytoscape element that are not equal and returns the two nodes
    that are included in the tree_dict but not in the cyto_elements.
    It also returns the edges to connect them to the existing nodes
    in the cystoscape elements.

    :param tree_nodes: List with all tree nodes
    :param cyto_elements: Ids of all nodes in the cytoscape element
    :return:
        nodes: a list with the two new nodes
        edges: a list with the two new edges
    """

    tree_nodes_id = list()
    for node in tree_nodes:
        tree_nodes_id.append(node['id'])
    # Find the new nodes in tree that are not in cystoscape element
    new_nodes_ids = list()
    for id in tree_nodes_id:
        if id not in cyto_nodes_id:
            new_nodes_ids.append(id)
    # Generate the two new nodes for cytoscape element
    new_nodes = list()
    new_edges = list()
    for id in new_nodes_ids:
        node = next((node for node in tree_nodes if node['id'] == id), None)
        fatherNode = next((node for node in tree_nodes if node['leftChild'] == id or node['rightChild'] == id), None)

        element = {
            'group': 'nodes',
            'data': {'id': node['id'], 'label': node['id']},
        }
        for j, prob in enumerate(node['classPurity']):
            class_str = 'class_{0}'.format(j)
            element['data'][class_str] = prob

        new_nodes.append(element)
        # Build label information
        if fatherNode['leftChild'] == id:
            labelStr = fatherNode['chosenAttr'] + ' < ' + str(round(fatherNode['threshold'], 1))
            type = 'leftEdge'
        else:
            labelStr = fatherNode['chosenAttr'] + ' > ' + str(round(fatherNode['threshold'], 1))
            type = 'rightEdge'
        new_edges.append({
            'group': 'edges',
            'data': {'source': node['fatherId'], 'target': node['id'], 'label': labelStr},
            'selected': False,
            'selectable': False,
            'classes': type
        })

    return new_nodes, new_edges


def remove_nodes(tree_nodes, old_elements, cyto_nodes_id):
    """
    This function takes a list of nodes from the tree and the list of elements
    of a cytoscape object that are not equal and returns the new list of
    elements for the cystoscape object.
    The new list of elements will be shorter than the old one since we are
    going to remove all the elements in the list that are no longer on
    the dictionary tree


    :param tree_nodes: List with all tree nodes
    :param old_elements: list with all the elements of the cytoscape object
    :param cyto_nodes_id: list with all ids of the cytoscape nodes elements
    :return:
        new_elements: new list with the resulting elements
    """
    tree_nodes_id = list()
    for node in tree_nodes:
        tree_nodes_id.append(node['id'])
    # Find the nodes in the cystoscape element that are no longer on the tree dictionary
    removed_nodes_ids = list()
    for id in cyto_nodes_id:
        if id not in tree_nodes_id:
            removed_nodes_ids.append(id)
    new_elements = list()
    for element in old_elements:
        if element['group'] == 'nodes':
            if element['data']['id'] not in removed_nodes_ids:
                new_elements.append(element)
        elif element['group'] == 'edges':
            if element['data']['target'] not in removed_nodes_ids:
                new_elements.append(element)

    return new_elements


# Callback for generating stylesheet when creating tree-dict
@app.callback(Output('tree_cyto', 'stylesheet'),
              [Input('tree-dict', 'children')],
              [State('tree_cyto', 'stylesheet')])
def generating_stylesheet(tree_jsonified, current_stylesheet):
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        if len(tree['nodes']) == 1:
            classes = tree['data']['classes']
            color_scale_rgba = tree['data']['color_scale']
            color_scale = list()
            for color in color_scale_rgba:
                rgb_tuple = rgba_to_rgb(color)
                color_scale.append('rgb' + str(rgb_tuple))
            new_styles = [{
                'selector': 'node',
                'style': {
                    'width': '60px',
                    'height': '60px',
                    'pie-size': '80%'
                }
                },
                {
                    'selector': '.leftEdge',
                    'style': {
                        'content': 'data(label)',
                        'edge-text-rotation': 'autorotate',
                        'text-margin-x': -10

                    }
                },
                {
                    'selector': '.rightEdge',
                    'style': {
                        'content': 'data(label)',
                        'edge-text-rotation': 'autorotate',
                        'text-margin-x': 10

                    }
                },
                {
                    'selector': '.openNode',
                    'style': {
                        'background-color': 'rgb(255,0,0)',
                        'line-color': 'rgb(255,0,0)',
                        'target-arrow-color': 'rgb(255,0,0)',
                        'source-arrow-color': 'rgb(255,0,0)',
                        'border-color': 'rgb(255,0,0)'
                    }
                },
                {
                    'selector': '.closedNode',
                    'style': {
                        'background-color': 'rgb(50,205,50)',
                        'line-color': 'rgb(50,205,50)',
                        'target-arrow-color': 'rgb(50,205,50)',
                        'source-arrow-color': 'rgb(50,205,50)',
                        'border-color': 'rgb(50,205,50)'
                    }
                },
                {
                'selector': ':selected',
                'style': {
                    'background-color': 'rgb(80,80,80)',
                    'line-color': 'rgb(80,80,80)',
                    'target-arrow-color': 'rgb(80,80,80)',
                    'source-arrow-color': 'rgb(80,80,80)'
                  }

                }]
            for j, cat in enumerate(classes):
                str_color = 'pie-{0}-background-color'.format(j+1)
                str_size = 'pie-{0}-background-size'.format(j+1)
                new_styles[0]['style'][str_color] = color_scale[j]
                new_styles[0]['style'][str_size] = 'mapData(class_{0}, 0, 1, 0, 100)'.format(j)

            return new_styles
        else:
            return current_stylesheet
    else:
        return current_stylesheet


# Callback for including the title of the section when generating a new tree
@app.callback(Output('tree-title', 'children'),
              [Input('tree-dict', 'children')])
def setTreeTitle(tree_jsonified):
    if tree_jsonified is not None:
        title = html.H6("Decision Tree",
                    style={
                    'textAlign': 'center'
                        }
                    )
        return title


# Callback for generating/deleting graph elements
@app.callback(Output('tree_cyto', 'elements'),
              [Input('tree-dict', 'children')],
              [State('tree_cyto', 'elements')])
def udpate_tree_elements(tree_jsonified, old_elements):
    # First case: Root Node to be represented when tree is just created
    # or needs to be recreated when the tree has been redone due to
    # algorithms main parameters change
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        # First case: Root Node to be represented when tree is just created
        # or needs to be recreated when the tree has been redone due to
        # algorithms main parameters change
        if len(old_elements) == 0:
            if len(tree['nodes']) != 1:
                print("ERROR: Unexpected situation when generating cytoscape object for a new tree")
            else:
                root = tree['nodes'][0]
                elements = [{
                    'group': 'nodes',
                    'data': {'id': root['id'], 'label': root['id']},
                    'selected': False,
                    'classes': 'openNode'
                }]
                for j, prob in enumerate(tree['nodes'][0]['classPurity']):
                    class_str = 'class_{0}'.format(j)
                    elements[0]['data'][class_str] = prob
                return elements
        else:
            # Second case:
            #   - the tree has been recreated due to changes in main parameters
            #   - all the children of the Root Node have been deleted
            # In any of the two cases, we rebuild the draw with only the root node
            if len(tree['nodes']) == 1:
                root = tree['nodes'][0]
                elements = [{
                    'group': 'nodes',
                    'data': {'id': root['id'], 'label': root['id']},
                    'selected': False,
                    'classes': 'openNode'
                }]
                for j, prob in enumerate(tree['nodes'][0]['classPurity']):
                    class_str = 'class_{0}'.format(j)
                    elements[0]['data'][class_str] = prob
                return elements
            # Rest of cases: Tree and cytoscape representation exist and need to be updated
            else:
                # Get tree nodes ids
                tree_nodes_id = list()
                for node in tree['nodes']:
                    tree_nodes_id.append(node['id'])
                # Get cytoscape tree nodes ids
                cyto_nodes_id = list()
                for element in old_elements:
                    if element['group'] == 'nodes':
                        cyto_nodes_id.append(element['data']['id'])
                # Compare both lists
                if len(tree_nodes_id) != len(cyto_nodes_id):
                    # Thrid case: Two nodes have been just added to the dictionary tree
                    if len(cyto_nodes_id) + 2 == len(tree_nodes_id):
                        new_nodes, new_edges = add_two_nodes(tree['nodes'], cyto_nodes_id)
                        elements = old_elements + new_nodes + new_edges
                        elements = check_elements(tree, elements)
                        return elements
                        # todo: COMPROBAR SI EL NODO ES HOJA O SI TIENE HIJOS
                        # return old_elements + new_nodes + new_edges
                    # Fourth case: Nodes have been deleted from the dictionary tree
                    elif len(cyto_nodes_id) > len(tree_nodes_id):
                        new_elements = remove_nodes(tree['nodes'], old_elements, cyto_nodes_id)
                        new_elements = check_elements(tree, new_elements)
                        return new_elements
                    else:
                        print("ERROR: Unexpected situation to update cytoscape object")
                else:
                    updated_elements = check_elements(tree, old_elements)
                    return updated_elements
    else:
        # Fith case: Tree has been removed or does not exist
        return []


# ==== CALLBACKS FOR CLASSIFICATION CONTROLS ====

@app.callback(Output('class-power-button', 'disabled'),
              [Input('tree-dict', 'children')])
def activateClassifier(tree_jsonified):
    if tree_jsonified is not None:
        tree = json.loads(tree_jsonified)
        result = isTreeClassifier(tree)
        if result:
            return False

    return True


@app.callback(Output('criterion-dropdown', 'disabled'),
              [Input('class-power-button', 'disabled')])
def disableCriterion(powerDisabled):
    if not powerDisabled:
        return False
    else:
        return True


@app.callback(Output('max-depth', 'disabled'),
              [Input('class-power-button', 'disabled')])
def disableCriterion(powerDisabled):
    if not powerDisabled:
        return False
    else:
        return True


@app.callback(Output('class-power-button', 'on'),
              [Input('tree-dict', 'children')],
              [State('class-power-button', 'on')])
def setToOffClassifier(tree_jsonified, currentOn):
    # We need to control if current status is ON and
    # the tree has changed in a way that now it cannot
    # be used for classification (i.e. one terminating node
    # is no longer leaf)
    if currentOn:
        if tree_jsonified is not None:
            tree = json.loads(tree_jsonified)
            result = isTreeClassifier(tree)
            if not result:
                return False


@app.callback(Output('classification_dict', 'children'),
              [Input('class-power-button', 'on')],
              [State('dataframe', 'children'),
               State('tree-dict', 'children'),
               State('train_test_df', 'children'),
               State('lda_dict', 'children'),
               State('criterion-dropdown', 'value'),
               State('max-depth', 'value')])
def execClassification(onStatus, df_jsonified, tree_jsonified, train_test_json, jsonified_data, criterion, max_depth):
    if onStatus:
        # Execute classification with own built decision tree
        # ###################################################
        # We only need to use the test Dataframe
        train_test = json.loads(train_test_json)
        X_test = pd.read_json(train_test['X_test'], orient='split')
        X_test_index = X_test.index.tolist()
        y_true, y_pred, classes, labels, own_dot_data = executeClassification(tree_jsonified, df_jsonified, X_test_index)

        # Execute classification with Sklearn DecisionTreeClassifier
        # ##########################################################
        # We need to use train and test Dataframes
        lda_dict = json.loads(jsonified_data)['lda']
        y_true_sklearn, y_pred_sklearn, classes_sklearn, labels_sklearn, sk_dot_data = \
            executeClassifictaionSklearn(lda_dict, criterion, max_depth)

        # Debug
        if len(classes) != len(classes_sklearn):
            print('Opss... something went wrong during classification...')
        else:
            for i in range(len(classes)):
                if classes[i] != classes_sklearn[i]:
                    print('Opss... something went wrong during classification...')
        # End Debug

        classification_dict = {
            'built_tree': {
                'y_true': y_true,
                'y_pred': y_pred,
            },
            'sklearn_tree': {
                'y_true': y_true_sklearn,
                'y_pred': y_pred_sklearn,
            },
            'classes': classes,
            'labels': labels,
            'sk_dot': sk_dot_data,
            'own_dot': own_dot_data
        }

        return json.dumps(classification_dict)
    else:
        return None


# ==== CALLBACKS FOR TREE RESULTS TABLES ====
@app.callback(Output('sklearn-card', 'children'),
             [Input('classification_dict', 'children')])
def showSklearnTable(class_dict_jsonified):
        if class_dict_jsonified is None:
            return []
        else:
            class_dict = json.loads(class_dict_jsonified)
            labels = class_dict['labels']
            sk_tree_report = classification_report(class_dict['sklearn_tree']['y_true'],
                                                   class_dict['sklearn_tree']['y_pred'],
                                                   target_names=labels,
                                                   output_dict=True)
            sk_tree_data = list()
            for item in labels:
                data_item = {'Class Label': item,
                             'Precision': round(sk_tree_report[item]['precision'], 4),
                             'Recall': round(sk_tree_report[item]['recall'], 4),
                             'F1-Score': round(sk_tree_report[item]['f1-score'], 4),
                             'Support': round(sk_tree_report[item]['support'], 4)}
                sk_tree_data.append(data_item)

            return [
                dash_table.DataTable(
                    id='own-table',
                    columns=
                    [
                        {'name': 'Class Label', 'id': 'Class Label'},
                        {'name': 'Precision', 'id': 'Precision'},
                        {'name': 'Recall', 'id': 'Recall'},
                        {'name': 'F1-Score', 'id': 'F1-Score'},
                        {'name': 'Support', 'id': 'Support'}
                    ],
                    data=sk_tree_data,
                )
            ]


@app.callback(Output('own-card', 'children'),
             [Input('classification_dict', 'children')])
def showOwnTable(class_dict_jsonified):
        if class_dict_jsonified is None:
            return []
        else:
            class_dict = json.loads(class_dict_jsonified)
            labels = class_dict['labels']
            own_tree_report = classification_report(class_dict['built_tree']['y_true'],
                                                   class_dict['built_tree']['y_pred'],
                                                   target_names=labels,
                                                   output_dict=True)
            own_tree_data = list()
            for item in labels:
                data_item = {'Class Label': item,
                             'Precision': round(own_tree_report[item]['precision'], 4),
                             'Recall': round(own_tree_report[item]['recall'], 4),
                             'F1-Score': round(own_tree_report[item]['f1-score'], 4),
                             'Support': round(own_tree_report[item]['support'], 4)}
                own_tree_data.append(data_item)

            return [
                dash_table.DataTable(
                    id='own-table',
                    columns=
                    [
                        {'name': 'Class Label', 'id': 'Class Label'},
                        {'name': 'Precision', 'id': 'Precision'},
                        {'name': 'Recall', 'id': 'Recall'},
                        {'name': 'F1-Score', 'id': 'F1-Score'},
                        {'name': 'Support', 'id': 'Support'}
                    ],
                    data=own_tree_data,
                )
            ]


@app.callback(Output('export-button', 'disabled'),
              [Input('own-card', 'children'),
               Input('sklearn-card', 'children')])
def disable_export_button(own_card, sklearn_card):
    if len(own_card) == 0 or len(sklearn_card) == 0:
        return True
    else:
        return False


@app.callback(
    Output("modal_export", "is_open"),
    [Input("export-button", "n_clicks"),
     Input("close_export", "n_clicks")],
    [State("modal_export", "is_open")],
)
def toggle_modal_export(n1, n2, is_open):

    if n1 or n2:
        return not is_open
    return is_open


@app.callback(Output('export-button', 'children'),
              [Input('export-button', 'n_clicks')],
              [State('classification_dict', 'children')])
def disable_export_button(n_clicks, classificaiton_dict_jsonified):
    if classificaiton_dict_jsonified is not None:
        classification_dict = json.loads(classificaiton_dict_jsonified)
        sk_dot = classification_dict['sk_dot']
        own_dot = classification_dict['own_dot']
        sk_graph = pydotplus.graph_from_dot_data(sk_dot)
        sk_graph = adapt_scikit_learn_colormap(sk_graph)
        own_graph = pydotplus.graph_from_dot_data(own_dot)
        sk_graph.write_pdf("sk_tree.pdf")
        own_graph.write_pdf("own_tree.pdf")

    return 'Export'


if __name__ == "__main__":
    app.run_server(debug=False, host="127.0.0.1")
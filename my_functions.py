import base64
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import numpy as np
from scipy import linalg
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import io
import webcolors as wc
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import json

import pydotplus
import ast
from c45_tree import rgba_to_rgb


def make_axis(axis, columns):
    columns.remove('index')
    axis_lines = []
    axis_labels = []
    for j, col in enumerate(columns):
        axis_lines.append(
            {
                'type': 'line',
                'xref': 'x1',
                'yref': 'y1',
                'x0': 0,
                'y0': 0,
                'x1': axis[0, j],
                'y1': axis[1, j],
                'line': {
                    'color': 'red',
                    'width': 2,
                },
            }
        )
        axis_labels.append(
            {
                'x': 1.1 * axis[0, j],
                'y': 1.1 * axis[1, j],
                'xref': 'x1',
                'yref': 'y1',
                'text': col,
                'showarrow': False,
                'font': {
                    'size': 14,
                    'color': '#000000'
                }
            }
        )
    return axis_lines, axis_labels


def parse_file(contents, filename, separator):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=separator)
    except Exception as e:
        print(e)
        return None
    return df


def split_train_test(df_full, targetFeature, trainValue, seed, full_train=False):
    y = df_full[targetFeature]
    X = df_full.drop(targetFeature, axis=1)
    X = X.apply(pd.to_numeric, errors='coerce')
    scaler = StandardScaler()

    if full_train:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - (trainValue / 100), random_state=seed)
    X_train_index = X_train['index']
    X_test_index = X_test['index']

    X_train_scaled = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    X_test.drop(['index'], axis=1, inplace=True)
    X_train.drop(['index'], axis=1, inplace=True)

    X_test.insert(len(X_test.columns), 'index', X_test_index)
    X_train.insert(len(X_train.columns), 'index', X_train_index)

    return X_train, X_test, y_train, y_test


def get_color_scale(lenght):
    # colors = px.colors.qualitative.Plotly
    colors = px.colors.qualitative.Dark24
    myColors = colors[:lenght]
    finalColors = list()
    for color in myColors:
        if not color.startswith('rgb'):
            finalColors.append('rgb' + str(wc.hex_to_rgb(color)))
        else:
            finalColors.append(color)
    return finalColors


def lda_2_classes_jittering(X_train, y_train):
    lda = LDA(n_components=2)
    X_train.apply(pd.to_numeric, errors='coerce')
    lda.fit(X_train, y_train)

    axis = lda.scalings_.T
    axis_descomposed = axis / linalg.norm(axis, 2)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train, y_train)
    components_pca = pca.components_

    b = [components_pca[0]]
    b = np.asarray(b)

    first_component = (axis_descomposed.T * b) + (1 / 10000)
    second_component = (axis_descomposed.T * axis_descomposed) + (1 / 10000)
    third_component = first_component / second_component
    forth_component = third_component * axis_descomposed

    b = b - forth_component
    b = b / linalg.norm(b, 2)

    transformation_matrix = np.array([axis_descomposed[0], b[0]])

    return transformation_matrix


def lda(X_train, X_test, y_train, y_test, features, seed, num_classes, jittering=True):
    id_train = X_train['index']
    id_test = X_test['index']
    X_train = X_train.drop('index', axis=1)
    X_test = X_test.drop('index', axis=1)
    y_train_index_values = y_train.index.values

    if num_classes is None:
        num_classes = len(y_train.unique())

    # Calculo los ejes del LDA
    if (num_classes == 2) & jittering:
        axis = lda_2_classes_jittering(X_train, y_train)
        Xr_train = np.dot(X_train, axis.T)
        Xr_test = np.dot(X_test, axis.T)
    else:
        lda = LDA(n_components=2)
        lda.fit(X_train, y_train)
        axis = lda.scalings_.T
        Xr_train = lda.transform(X_train)
        Xr_test = lda.transform(X_test)

    if num_classes != 2 or jittering:
        Xr_train_df = pd.DataFrame(Xr_train, index=y_train_index_values, columns=['x', 'y'])
        Xr_test_df = pd.DataFrame(Xr_test, index=id_test, columns=['x', 'y'])
    else:
        Xr_train_df = pd.DataFrame(Xr_train, index=y_train_index_values, columns=['x'])
        Xr_test_df = pd.DataFrame(Xr_test, index=id_test, columns=['x'])

    Xr_test_df = Xr_test_df.reset_index(level=0, drop=False)
    Xr_train_df = Xr_train_df.reset_index(level=0, drop=False)

    if num_classes != 2 or jittering:
        Xr_train_df = Xr_train_df[['x', 'y', 'index']]
        Xr_test_df = Xr_test_df[['x', 'y', 'index']]
    else:
        Xr_train_df = Xr_train_df[['x', 'index']]
        Xr_test_df = Xr_test_df[['x', 'index']]

    Xr_train_df = Xr_train_df.values
    Xr_test_df = Xr_test_df.values

    return Xr_train_df, Xr_test_df, axis, num_classes


def lda_2_classes_jittering(X_train, y_train):
    lda = LDA(n_components=2)
    X_train.apply(pd.to_numeric, errors='coerce')
    lda.fit(X_train, y_train)

    axis = lda.scalings_.T
    axis_descomposed = axis / linalg.norm(axis, 2)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train, y_train)
    components_pca = pca.components_

    b = [components_pca[0]]
    b = np.asarray(b)

    first_component = (axis_descomposed.T * b) + (1 / 10000)
    second_component = (axis_descomposed.T * axis_descomposed) + (1 / 10000)
    third_component = first_component / second_component
    forth_component = third_component * axis_descomposed

    b = b - forth_component
    b = b / linalg.norm(b, 2)

    transformation_matrix = np.array([axis_descomposed[0], b[0]])

    return transformation_matrix


def make_figure(lda_json, original_classes, jittering=True):
    data_dict = json.loads(lda_json)
    lda_dict = data_dict['lda']
    color_scale = data_dict['color_scale']
    # TODO: Añadir como caso normal si se muestra con jittering
    # Normal case: more than 2 classes of target feature
    if lda_dict['num_classes'] != 2 or jittering:

        axis_param = dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            ticks='inside',
            ticklen=4,
            mirror='ticks',
        )

        shapes = []
        annotations = []
        traces = []
        layout = go.Layout()
        error_legend = False

        layout['xaxis'] = dict(axis_param, domain=[0, 1], anchor='y')
        layout['yaxis'] = dict(axis_param, domain=[0, 1], anchor='x')

        features = lda_dict['features']

        Xr_train = pd.read_json(lda_dict['Xr_train'], orient='split')
        Xr_train = Xr_train.values

        X_train = pd.read_json(lda_dict['X_train'], orient='split').values
        X_train_not_scaled = pd.read_json(lda_dict['X_train_not_scaled'], orient='split')
        y_train = pd.read_json(lda_dict['y_train'], orient='split', typ='series')

        # ¿Por qué?
        represented_train = Xr_train[:, 2]
        X_train_not_scaled = X_train_not_scaled.loc[represented_train]

        features_text = X_train_not_scaled.columns.tolist()
        X_train_not_scaled = X_train_not_scaled.values

        axis = pd.read_json(lda_dict['axis'], orient='split').values

        target_feature = lda_dict['target_feature']

        zeroline = []
        zeroline.append(
            {
                'type': 'circle',
                'xref': 'x1',
                'yref': 'y1',
                'x0': -1,
                'y0': -1,
                'x1': 1,
                'y1': 1,
                'line': {
                    'color': 'rgb(125, 125, 125)',
                    'width': 1,
                    'dash': 'dot',
                },
            }
        )

        zeroline.append(
            {
                'type': 'line',
                'xref': 'x1',
                'yref': 'y1',
                'x0': -1,
                'y0': 0,
                'x1': 1,
                'y1': 0,
                'line': {
                    'color': 'rgb(125, 125, 125)',
                    'width': 1,
                    'dash': 'dot',
                },
            }
        )

        zeroline.append(
            {
                'type': 'line',
                'xref': 'x1',
                'yref': 'y1',
                'x0': 0,
                'y0': -1,
                'x1': 0,
                'y1': 1,
                'line': {
                    'color': 'rgb(125, 125, 125)',
                    'width': 1,
                    'dash': 'dot',
                },
            }
        )

        shapes.extend(zeroline)
        title_text = 'LDA Projection'
        title = go.layout.Annotation(
            x=0.5,
            y=1,
            xanchor='center',
            yanchor='bottom',
            xref='paper',
            yref='paper',
            text=title_text,
            showarrow=False,
            font={'size': 16}
        )

        axis_lines, axis_labels = make_axis(axis, features)
        shapes.extend(axis_lines)
        annotations.extend(axis_labels)
        annotations.append(title)

        classes = sorted(y_train.unique())

        for j, cat in enumerate(classes):

            hover_trace_train = []
            data_cat = X_train_not_scaled[y_train == cat]

            for row in data_cat:
                hover_trace_train.append("-- Train Observation -- <br>")
                hover_trace_train[-1] += "Category: {} <br>".format(cat)
                hover_trace_train[-1] += "-- Original Features -- <br>"

                for pos, element in enumerate(features_text):
                    if pos % 3 == 0:
                        hover_trace_train[-1] += "{}: {:.2f}  <br>".format(element, row[pos])
                    else:
                        hover_trace_train[-1] += "{}: {:.2f} . ".format(element, row[pos])

            if original_classes is None:
                trace_train = go.Scattergl(
                    x=Xr_train[y_train == cat][:, 0],
                    y=Xr_train[y_train == cat][:, 1],
                    xaxis='x',
                    yaxis='y',
                    mode='markers',
                    marker=dict(
                        color='rgba' + color_scale[j][3:-1] + ', 0.8)',
                        line=dict(width=1, color=color_scale[j])
                    ),
                    ids=Xr_train[y_train == cat][:, 2],
                    name=str(cat),
                    legendgroup=str(cat),
                    showlegend=True,
                    hoverinfo='text',
                    hovertext=hover_trace_train,
                )
            else:
                index = original_classes.index(cat)
                trace_train = go.Scattergl(
                    x=Xr_train[y_train == cat][:, 0],
                    y=Xr_train[y_train == cat][:, 1],
                    xaxis='x',
                    yaxis='y',
                    mode='markers',
                    marker=dict(
                        color='rgba' + color_scale[index][3:-1] + ', 0.8)',
                        line=dict(width=1, color=color_scale[index])
                    ),
                    ids=Xr_train[y_train == cat][:, 2],
                    name=str(cat),
                    # name = Xr_train[y_train==cat][:,2],
                    legendgroup=str(cat),
                    showlegend=True,
                    hoverinfo='text',
                    # text=hover_trace_train
                    hovertext=hover_trace_train,
                )

            traces.append(trace_train)

        layout['annotations'] = annotations
        layout['shapes'] = shapes
        layout['margin'] = go.layout.Margin(l=40, r=40, t=60, b=40)
        layout['hovermode'] = 'closest'

        figure = go.Figure(
            data=traces,
            layout=layout
        )
        return figure
    # Special case for target feature having only two classes
    # Displot will be used instead of scatter plot
    else:
        Xr_train = pd.read_json(lda_dict['Xr_train'], orient='split')

        # Preparing for rug text
        X_train_not_scaled = pd.read_json(lda_dict['X_train_not_scaled'], orient='split')

        represented_train = Xr_train.values[:, 1]
        X_train_not_scaled = X_train_not_scaled.loc[represented_train]

        features_text = X_train_not_scaled.columns.tolist()
        X_train_not_scaled = X_train_not_scaled.values
        # End of preparing for rug text

        cols = Xr_train.columns.values
        Xr_train.set_index(cols[-1], inplace=True)
        y_train = pd.read_json(lda_dict['y_train'], orient='split', typ='series')

        index_per_class = list()
        class_labels = list()
        for item in [i for i in sorted(list(y_train.unique()))]:
            index_per_class.append(y_train[y_train == item].index.values)
            class_labels.append(str(item))

        hist_data = list()
        for indexes in index_per_class:
            hist_data.append(Xr_train.loc[indexes].values.flatten())

        # Protect against low dimensional nodes. Every class must have more than 1 value
        final_hist = list()
        final_class_labels = list()
        for i in range(len(hist_data)):
            if len(hist_data[i]) > 1:
                final_hist.append(hist_data[i])
                final_class_labels.append(class_labels[i])
        final_class_labels.sort()

        colors = list()
        if original_classes is not None:
            classes = sorted([int(i) for i in final_class_labels])
            for j, cat in enumerate(classes):
                index = original_classes.index(cat)
                colors.append('rgba' + color_scale[index][3:-1] + ', 0.8)')
        else:
            classes = sorted(y_train.unique())
            for j, cat in enumerate(classes):
                colors.append('rgba' + color_scale[j][3:-1] + ', 0.8)')

        # For hover info
        rug_text = []
        for j, cat in enumerate(final_class_labels):

            hover_text = []
            data_cat = X_train_not_scaled[y_train == int(cat)]

            for row in data_cat:
                hover_text.append("-- Train Observation -- <br>")
                hover_text[-1] += "Category: {} <br>".format(cat)
                hover_text[-1] += "-- Original Features -- <br>"

                for pos, element in enumerate(features_text):
                    if pos % 3 == 0:
                        hover_text[-1] += "{}: {:.2f}  <br>".format(element, row[pos])
                    else:
                        hover_text[-1] += "{}: {:.2f} . ".format(element, row[pos])

            rug_text.append(hover_text)

        # End of hover info generation
        figure = ff.create_distplot(final_hist, final_class_labels, colors=colors, rug_text=rug_text)
        figure.update_layout(title='LDA Projection (1-D) of two classes: Distribution Plot', title_x=0.5)
        return figure


def executeClassifictaionSklearn(lda_dict, criterion, max_depth):
    """
    This function takes lda_dict as input and generates and execute a classification using
    Sklearn Decission Tree.

    :param lda_dict: Dict containing: X_train, X_test, X_train_not_scaled, X_test_not_scaled, Xr_train, Xr_test,
                     y_train, y_test, axis, features and target_feature
    :param criterion: Criterion for DecisionTreeClassifier function: gini or entropy
    :param max_depth: Maximum depth for tree generation for DecisionTreeClassifier function
    :return: Tuple containing:
                - y_true: list with the real classes for the test observations
                - y_pred: list with the predicted classes for the test observations
                - classes: list with the different classes
                - labels: list with the labels for the classes
                - dot string: string representing the decision tree in DOT format
    """
    X_train_with_index = pd.read_json(lda_dict['X_train'], orient='split')
    X_train = X_train_with_index.drop(columns=['index'])

    X_test_with_index = pd.read_json(lda_dict['X_test'], orient='split')
    X_test = X_test_with_index.drop(columns=['index'])

    targetFeature = lda_dict['target_feature']

    y_train = pd.read_json(lda_dict['y_train'], orient='split', typ='series')
    y_test = pd.read_json(lda_dict['y_test'], orient='split', typ='series')

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Generate dot string
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=list(X_train.columns),
                                    # filled=False,
                                    filled=True,
                                    rounded=True,
                                    proportion=True)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # colormap = ["rgba(99, 110, 250, 1)", "rgba(239, 85, 59, 1)", "rgba(0, 204, 150, 1)"]
    #
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # nodes = graph.get_node_list()
    #
    # leafs = {n.get_name(): True for n in graph.get_nodes()}
    # for e in graph.get_edge_list():
    #     leafs[e.get_source()] = False
    # del leafs['node']
    # del leafs['edge']
    # leafs = {key: val for key, val in leafs.items() if val is True}
    #
    # for node in nodes:
    #     if node.get_name() not in ('node', 'edge'):
    #         if node.get_name() in leafs:
    #             label = node.get('label')
    #             a = '[' + label.split('[')[1]
    #             a = a.replace('"', '')
    #             b = ast.literal_eval(a)
    #             index = b.index(max(b))
    #             color = colormap[index]
    #             color = rgba_to_rgb(color)
    #             color = wc.rgb_to_hex(color)
    #             node.set_fillcolor(color)
    #         else:
    #             node.set_fillcolor('#FFFFFF')
    #
    # # for leaf in leafs:
    # #     node = nodes[leaf]
    # #     label = node.get('label')
    # #     a = '[' + label.split('[')[1]
    # #     a = a.replace('"', '')
    # #     b = ast.literal_eval(a)
    # #     index = b.index(max(b))
    # #     color = colormap[index]
    #
    #
    # graph.write_png('test_tree.png')

    # Get classes and classes labels
    classes = sorted(y_train.unique())
    labels = ['class_{}'.format(item) for item in classes]

    return y_test.tolist(), y_pred.tolist(), classes, labels, dot_data


def adapt_scikit_learn_colormap(graph):
    # colormap = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
    #             '#FF97FF', '#FECB52']

    colormap = px.colors.qualitative.Dark24

    nodes = graph.get_node_list()

    leafs = {n.get_name(): True for n in graph.get_nodes()}
    for e in graph.get_edge_list():
        leafs[e.get_source()] = False
    del leafs['node']
    del leafs['edge']
    leafs = {key: val for key, val in leafs.items() if val is True}

    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            if node.get_name() in leafs:
                label = node.get('label')
                a = '[' + label.split('[')[1]
                a = a.replace('"', '')
                a = a.replace('\\n', ',')
                b = ast.literal_eval(a)
                index = b.index(max(b))
                color = colormap[index]
                node.set_fillcolor(color)
            else:
                node.set_fillcolor('#FFFFFF')

    return graph


def check_elements(tree, elements):
    open_nodes_list = []
    closed_nodes_list = []
    for node in tree['nodes']:
        if node['isLeaf'] is False and node['hasChildren'] is False:
            open_nodes_list.append(node['id'])
        if node['isLeaf'] is True:
            closed_nodes_list.append(node['id'])

    for element in elements:
        if element['group'] == 'nodes':
            if element['data']['id'] in open_nodes_list:
                element['classes'] = 'openNode'
            elif element['data']['id'] in closed_nodes_list:
                element['classes'] = 'closedNode'
            else:
                element['classes'] = ''

    return elements

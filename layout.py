import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_cytoscape as cyto

# ==== NAV BAR WITH LOGO AND INFO MODAL BUTTON ====

INFO_MODAL = "This tool is designed to help users to manually build a decision tree classifier based on their domain knowledge.\n\n" \
"The idea behind this tool is described in the research article: 'Visually guided classification trees for analyzing chronic patients' " \
"(Cristina Soguero-Ruiz, Inmaculada Mora-Jiménez, Miguel A. Mohedano-Munoz, Manuel Rubio-Sanchez, Pablo de Miguel-Bohoyo, and Alberto Sanchez).\n\n" \
"The tool has two main parts. In the upper one, on the left side there are two tabs, one to load the CSV file, chose the target class, " \
"the features to use and the percentage of the dataframe to be used as training set. On the right tab, we have the controls to build the decision tree. " \
"One can chose the feature to partition a tree node and the tool will find the threshold value (using C4.5 algorithm) so the information gain is " \
"maximized. The buttons to partition a node (once the feature is chosen) and to remove the children of the selected node, are " \
"both located below chosen parameter drop-down control. " \
"Besides, the main characteristics of each node are shown and finally one button to force a node to be a leaf (and another one to revert back that action).\n\n" \
"One node is a leaf if its entropy is zero (probability of one class is 1) or if the user forces the node to be a leaf. " \
"Only when all the nodes without children are leaf, classification can be executed.\n" \
"In the bottom part of the tool, we have the classification controls. On the left side, one can tune the main parameters " \
"for the Sklearn decision tree classifier; criterion (either entropy or gini index) and tree maximum depth. Then, by pressing " \
"the On/Off button, one can activate the classifier, so the dataset is used to automatically generate a decision tree classifier " \
"by Sklearn DecisionTreeClassifier class on one hand, and on the other, using the manually built decision tree generate our own classifier. " \
"Using both of them, and the test part of the dataset, classification is done and both models are compared using Sklearn metrics " \
"module and classification_report function.\n\n" \
"Finally, both decision trees can be exported to PDF format using Graphviz by means of Export button."

MODAL = html.Div(
    [
        dbc.Button("INFO", id="open"),
        dbc.Modal(
            [
                dbc.ModalHeader("Visually Guided Decision Tree Builder"),
                dbc.ModalBody(INFO_MODAL),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="modal",
            size='lg',
            scrollable=True,
            style={'white-space': 'pre-line'}
        ),
    ]
)

NAVBAR = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    # dbc.Col(
                    #     html.A([
                    #         html.Img(src='./assets/dash-logo-new.png',
                    #                      className="logo",
                    #                      style={'height': '90%', 'width': '90%'})],
                    #             href="https://plot.ly"),
                    #     width=3),
                    dbc.Col(dbc.NavbarBrand("Visually Guided Decision Tree Builder", className="ml-5"),width=9),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
        dbc.Row(
            [
                dbc.Col(MODAL)
            ],
            className='ml-auto mr-5'
        )
    ],
    color="dark",
    dark=True,
    className="mb-2"
)

# ==== LEFT COLUMN: CONTROL TABS  ====

DATA_TAB = dbc.Tab(
    [
        html.H4(children="Upload and prepare data", className="display-5 mt-5"),
        html.Hr(className="my-2"),
        html.Label("Upload data", className="lead mb-2"),
        dcc.Upload(
            [
                dbc.Button(
                    "Upload",
                    color='primary',
                    className='mr-1',
                    style={
                        'display': 'inline-block'
                    }
                ),
                html.Label(
                    children=[],
                    style={
                        'flex-grow': 1,
                        'display': False
                    },
                    id='file-name',


                )
            ],
            id='upload-data',
            className='mb-2'
        ),
        dbc.FormGroup(
            [
                dbc.Label('Field Separator', width=5),
                dbc.Col(
                    dbc.Input(
                        type='text',
                        id='field-sep',
                        value=',',
                        maxLength=1,
                        bs_size='sm',
                    ),
                    width=2
                )
            ],
            row=True,
            className='mb-4'
        ),
        html.Label("Select target feature", className="lead"),
        dcc.Dropdown(
            id='target-feature',
            placeholder="Select Feature",
            disabled=True,
            style={
                'width': '100%'
            }
        ),
        html.Label("Selected features", className="lead mt-4 mb-2"),
        dcc.Dropdown(
                id='selected-features',
                placeholder="Select Features",
                multi=True,
                disabled=True,
                style={
                    'width': '100%'
                }
            ),
        html.Label("Train percentage", className="lead mt-4 mb-2"),
        dbc.Col([
            dbc.Row([
                daq.Knob(
                    id='train-knob',
                    max=100,
                    size=90,
                    value=0,
                    disabled=True
                )
            ],
            className='row justify-content-center'
            ),
            dbc.Row([
                daq.GraduatedBar(
                    id='train_bar',
                    showCurrentValue=True,
                    max=100,
                    value=0,
                )
            ],
            className='row justify-content-center'
            )
            ]
        )
    ],
    label="Data Preparation",
    tab_id="tab-1"
)

TREE_TAB = dbc.Tab(
    [
        html.H4(children="Decision tree controls", className="display-5 mt-5"),
        html.Hr(className="my-2"),
        html.Label("Chosen feature", className="lead mb-2"),
        dcc.Dropdown(
                id='chosenAttr-options',
                placeholder="Select Chosen Feature",
                disabled=True,
                style={
                    'width':'100%',
                },
            ),
        dcc.Checklist(
            options=[
                {'label': 'Apply jittering on 2 classes subsets', 'value': True}
            ],
            id='jittering_checklist'
        ),
        html.Label("Create/Delete nodes", className="lead mt-4 mb-2"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Create Node",
                        id='createNode-button',
                        color='primary',
                        n_clicks=0,
                        className='mr-1',
                        disabled=True,
                        style={
                            'display': 'inline-block'
                        }
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        "Delete Children",
                        id='deleteNode-button',
                        color='primary',
                        n_clicks=0,
                        className='mr-1',
                        disabled=True,
                        style={
                            'display': 'inline-block'
                        }
                    )
                )
            ]
        ),
        html.Label("Node data", className="lead mt-4 mb-2"),
        html.Div(
            id='text-area',
            children=[
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
                    html.Div("Entropy: ", style={'font-weight': 'bold', 'display': 'inline'}),
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
            ],
            style = {
                'height':'100%',
                'display': 'flex',
                'flex-flow': 'column'
            }
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Make Leaf",
                        id='forceLeaf-button',
                        color='primary',
                        n_clicks=0,
                        className='mr-1',
                        disabled=True,
                        style={
                            'display': 'inline-block'
                        }
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        "Reset Leaf",
                        id='resetLeaf-button',
                        color='primary',
                        n_clicks=0,
                        className='mr-1',
                        disabled=True,
                        style={
                            'display': 'inline-block'
                        }
                    )
                )
            ],
            className='mt-4'
        ),
    ],
    label="Decision Tree",
    tab_id="tab-2"
)

LEFT_COLUMN = dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        DATA_TAB,
                        TREE_TAB,
                    ],
                    id='card-tabs',
                    card=True,
                    active_tab='tab-1'
                )
            )
        ]
    )

# ==== RIGHT COLUMN: PLOTS  ====

table = html.A(children=[], id='table', style={'width':600, 'margin-left': '2%'})

RIGHT_COLUMN_1 = html.Div(
    [
        dcc.Graph(id='lda-plot', style=dict(visibility = 'hidden')),
    ],
)

tree_cyto = html.Div([
    html.H6(id='tree-title', children=[]),
    cyto.Cytoscape(
        id='tree_cyto',
        layout={
            'name': 'breadthfirst',
            'directed': 'true'
        },
        elements=[],
        maxZoom=2,
        stylesheet=[
            {'selector': 'node',
                'style': {
                    'width': '60px',
                    'height': '60px',
                    'pie-size': '80%', }
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

            },
        ]
    )
])


RIGHT_COLUMN_2 = html.Div([tree_cyto])

# ==== SECOND ROW: CLASSIFICATION CONTROLS  ====

CLASS_CONTROL_CARD_CONTENT = [
    dbc.CardHeader(
        html.H4('Classification Controls')
    ),
    dbc.CardBody([
        html.Label("Sklearn Decision Tree Parameters", className="lead mb-2"),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Label('Criterion'),
                ], width=5),
                dbc.Col([
                    dcc.Dropdown(
                        id='criterion-dropdown',
                        disabled=True,
                        clearable=False,
                        options=[
                            {'label': 'entropy', 'value': 'entropy'},
                            {'label': 'gini', 'value': 'gini'}
                        ],
                        value='entropy',
                        style={
                            'width': '100%'
                        }
                    )
                ], width=7)
            ],
            className='mb-2'
        ),
        dbc.FormGroup(
            [
                dbc.Col([
                    dbc.Label('Maximum Depth'),
                ], width=5),
                dbc.Col([
                    daq.NumericInput(
                        id='max-depth',
                        max=99,
                        min=1,
                        value=3,
                        disabled=True
                    )
                ], width=7)
            ],
            row=True,
            className='mb-2'
        ),
        html.Hr(className="my-2 mb-4"),
        html.Label('Execute & Compare Classificators', className='lead mb-2 row justify-content-center'),
        dbc.Row(
            [
                daq.PowerButton(
                    id='class-power-button',
                    size=80,
                    color='#FF5E5E',
                    disabled=True
                )
            ],
            className='row justify-content-center lead mb-4'
        ),
        html.Hr(className="my-2"),
        dbc.Row(
            [
                dbc.Col(
                    html.Label("Export Results", className="lead mb-2"),
                    className='column justify-content-center'
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            "Export",
                            id='export-button',
                            disabled=True,
                            color='primary',
                            n_clicks=0,
                            className='mr-1',
                            style={
                                'display': 'inline-block'
                            }
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Export Function"),
                                dbc.ModalBody("Sklearn and manually built trees exported to PDF format"),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="close_export", className="ml-auto")
                                ),
                            ],
                            id="modal_export",
                            scrollable=False
                        ),
                    ],
                    className='column justify-content-center'
                )
            ],
            className='row align-items-center lead mt-4 mb-2'
        ),
    ])
]

SKLEARN_CARD_CONTENT = [
    dbc.CardHeader(
        html.H4('Sklearn Decission Tree - Results')
    ),
    dbc.CardBody(
        id='sklearn-card',
        children= [],
        )
]

OWNTREE_CARD_CONTENT = [
    dbc.CardHeader(
        html.H4('Manual Decission Tree - Results')
    ),
    dbc.CardBody(
        id='own-card',
        children= [],
        )
]

LEFT_CARD = dbc.Card(CLASS_CONTROL_CARD_CONTENT, color='light')
RIGHT_CARD1 = dbc.Card(SKLEARN_CARD_CONTENT, color='light')
RIGHT_CARD2 = dbc.Card(OWNTREE_CARD_CONTENT, color='light')

# ==== BODY:  ====
# ==== ROW1: LEFT COLUMN + RIGHT COLUMN ====
# ==== ROW2: LEFT CARD + RIGHT CARD ====

BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=3),
                dbc.Col(
                    [
                        dbc.Row(RIGHT_COLUMN_1),
                        dbc.Row(table, className='row justify-content-center')
                    ],
                    md=5
                ),
                dbc.Col(RIGHT_COLUMN_2, md=4)
            ],
            style={"marginTop": 30},
            className='ml-0 mr-0 mb-0 mt-0'
        ),
        html.Hr(),
        dbc.Row([
            dbc.Col(
                [
                    LEFT_CARD
                ],
                md=3,
            ),
            dbc.Col(
                [
                    RIGHT_CARD1
                ],
            ),
            dbc.Col(
                [
                    RIGHT_CARD2
                ],
            ),
        ],
        style={"marginTop": 30},
        className='ml-0 mr-0 mb-0 mt-0'
        ),
        html.Div(id='seed', style={'display': 'none'}),
        html.Div(id='dataframe', style={'display': 'none'}),
        html.Div(id='df_to_use', style={'display': 'none'}),
        html.Div(id='train_test_df', style={'display': 'none'}),
        html.Div(id='lda_dict', style={'display': 'none'}),
        html.Div(id='lda_dict_node', style={'display': 'none'}),
        html.Div(id='tree-dict', style={'display': 'none'}),
        html.Div(id='classification_dict', style={'display': 'none'}),
    ],
    className="mt-12 ml-3",
    fluid=True
)

# ==== LAYOUT: NAV BAR + BODY  ====

app_layout = html.Div(children=[NAVBAR, BODY])
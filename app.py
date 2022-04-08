# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import re
from flask import Flask, request
import json
import numpy as np
import base64
from keras.models import load_model
from rdkit import Chem
from rdkit.Chem import Draw
import io

import sys

from scipy import rand
sys.path.insert(0, "Classifier")
import fingerprint_handler
import prediction_voting

server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = 'NP Classifier'

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server = app.server

# Loading in Models
pathway_path = ".\\Trained_Models\\pathway"
pathway_model = load_model(pathway_path)
super_path = ".\\Trained_Models\\superclass"
super_model = load_model(super_path)
class_path = ".\\Trained_Models\\class"
class_model = load_model(class_path)

ontology_dictionary = json.loads(open("./index_v2.json").read())

# NCI image
image_filename = './nci-logo-full.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

NAVBAR = dbc.Navbar(
    children=[
        dbc.NavbarBrand(
            html.Img(src="data:image/png;base64, " + str(encoded_image.decode("utf-8")), width="566px", style={"margin-left": "10px"}),
        )
    ],
    color="light",
    dark=False,
    sticky="top",
)

DASHBOARD = [
    dcc.Location(id='url', refresh=False),
    dbc.CardHeader(html.H5("NP Classifier (hosted locally)")),
    dbc.CardBody(
        [
            html.Div(id='version', children="Version - 1.Grant"),
            html.Br(),
            dbc.Textarea(className="mb-3", id='smiles_string', placeholder="Smiles Structure"),
            dcc.Loading(
                id="structure",
                children=[html.Div([html.Div(id="loading-output-5")])],
                type="default",
            ),
            dcc.Loading(
                id="classification_table",
                children=[html.Div([html.Div(id="loading-output-3")])],
                type="default",
            ),
            html.Hr(),
            dcc.Loading(
                id="usage_summary",
                children=[html.Div([html.Div(id="loading-output-323")])],
                type="default",
            )
        ]
    )
]

CONTRIBUTORS_DASHBOARD = [
    dbc.CardHeader(html.H5("Software Contributors")),
    dbc.CardBody(
        [
            "Hyunwoo Kim PhD - UC San Diego",
            html.Br(),
            "Mingxun Wang PhD - UC San Diego",
            html.Br(),
            "Grant McConachie B.Sc. - Oregon State University (Leidos)",
            html.Br(),
            html.Br(),
            html.H5("Citation"),
            html.A('Kim HW, Wang M, Leber CA, Nothias LF, Reher R, Kang KB, van der Hooft JJJ, Dorrestein PC, Gerwick WH, Cottrell GW. NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products. J Nat Prod. 2021 Oct 18. doi: 10.1021/acs.jnatprod.1c00399. Epub ahead of print. PMID: 34662515.', 
                    href="https://pubmed.ncbi.nlm.nih.gov/34662515/")
        ]
    )
]

BATCH_PROCESSING = [
    dbc.CardHeader(html.H5("Batch Processing")),
    dbc.CardBody(
        [
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File', style={'color': 'blue', 'cursor': 'pointer', 'text-decoration': 'underline'}),
                    ' (.xlsx or .xls)'
                ]),
                style={
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id="name-of-file", style={'margin-left': '8px', 'color': 'green'}),
            dcc.Input(id="smiles-column", type="number", placeholder="SMILES Column", style={'margin-left': '8px', 'margin-bottom': '10px'}),
            html.Button("Classify...", id="classify", style={'margin-left': '8px', 'margin-bottom': '10px'}),
            dbc.Spinner(html.Div("", id="spinner")),
            html.Div(dash_table.DataTable(
                id = 'dt1',
                export_format="csv",
                export_columns='all',
                style_cell={
                    'textAlign': 'left',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': 0
                },
                ),
            ),
            dbc.Container([
                html.Div(
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Incorrect Filetype")),
                        dbc.ModalBody("Please upload an excel file (.xls or .xlsx)."),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                        ),
                    ],
                    id="modal",
                    is_open=False,
                    )
                )
            ]),
            
        ]
    )
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(DASHBOARD)),], style={"marginTop": 30}),
        html.Br(),
        dbc.Row([dbc.Col(dbc.Card(CONTRIBUTORS_DASHBOARD)),]),
        html.Br(),
        dbc.Row([dbc.Col(dbc.Card(BATCH_PROCESSING)),]),
    ],
    className="mt-12",
)

app.layout = html.Div(children=[NAVBAR, BODY])


# Callback for batch processing, excel check
@app.callback(
    Output('name-of-file', 'children'),
    Output('modal', 'is_open'),
    Input('upload-data', 'filename'),
    Input('close', 'n_clicks'),
    State('modal', 'is_open')
)
def read_excel(filename, close_modal, modal_state):
    """Reads the excel file and makes sure that it can read the contents"""

    # Closing modal if we need
    if close_modal > 0 and modal_state == True:
        return "", False

    # Prevents app from updating if no file
    if filename is None:
        raise PreventUpdate

    # Deciding if the filetype is okay
    re_object = re.search('\.xls', filename)
    if re_object is not None:
        return filename, False
    else:
        return "", True


# Callback for batch processing, data output
@app.callback(
    Output('dt1', 'data'),
    Output('dt1', 'columns'),
    Output('smiles-column', 'style'),
    Output('spinner', 'children'),
    Input('classify', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('smiles-column', 'value')
)
def convert_smiles(n, contents, filename, smiles_column):
    """Takes the excel sheet and converts the smiles into classifications"""

    # Prevents app from updating if no file
    if filename is None:
        raise PreventUpdate

    # Catches if there is no smiles column entered
    re_object = re.search('\.xls', filename)
    if smiles_column == 0 or smiles_column is None and re_object is not None:
        return "", "", {'margin-left': '8px', 'border-color': 'red'}, ""

    elif n and re_object is not None:
        # check if xlsx
        re_object_xlsx = re.search('\.xlsx', filename)

        # Get the file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))

        # Put the smiles into the api
        export_df = pd.DataFrame() # empty datafram to fill
        smiles_column += 1
        for index, row in df.iterrows():
            smiles_string = str(row[smiles_column]) #TODO: FINISH THIS
            classification_dict = _process_full_classification(smiles_string)

            # If statenent putting blank spaces if no results come up
            if classification_dict['pathway_results'] == []:
                classification_dict['pathway_results'] = ['']
            if classification_dict['superclass_results'] == []:
                classification_dict['superclass_results'] = ['']
            if classification_dict['class_results'] == []:
                classification_dict['class_results'] = ['']

            # If more than one result
            if len(classification_dict['pathway_results']) > 1:
                classification_dict['pathway_results'] = [" / ".join(classification_dict['pathway_results'])]
            if len(classification_dict['superclass_results']) > 1:
                classification_dict['superclass_results'] = [" / ".join(classification_dict['superclass_results'])]
            if len(classification_dict['class_results']) > 1:
                classification_dict['class_results'] = [" / ".join(classification_dict['class_results'])]

            export_df = export_df.append(pd.DataFrame(
                [[smiles_string, classification_dict['pathway_results'][0], classification_dict['superclass_results'][0], classification_dict['class_results'][0], classification_dict['isglycoside']]], 
                columns=["SMILES", "Pathway", "Superclass", "Class", "Glycoside"],
            ))

        return [
                    export_df.to_dict("records"), 
                    [
                        {"name": "SMILES", "id": "SMILES"}, 
                        {"name": "Pathway", "id": "Pathway"}, 
                        {"name": "Superclass", "id": "Superclass"},
                        {"name": "Class", "id": "Class"},
                        {"name": "Glycoside", "id": "Glycoside"}
                    ], 
                    {'margin-left': '8px'}, 
                    ""
                ]
    
    else:
        raise PreventUpdate


# This enables parsing the URL to shove a task into the qemistree id
@app.callback(Output('smiles_string', 'value'),
              [Input('url', 'pathname')])
def display_page(pathname):
    # Otherwise, lets use the url
    if len(pathname) > 1:
        return pathname[1:]
    else:
        return "CC1C(O)CC2C1C(OC1OC(COC(C)=O)C(O)C(O)C1O)OC=C2C(O)=O"

# This function will rerun at any 
@app.callback(
    [Output('classification_table', 'children'), Output('structure', 'children')],
    [Input('smiles_string', 'value')],
)
def handle_smiles(smiles_string):
    classification_dict = _process_full_classification(smiles_string)
    #isglycoside, class_results, superclass_results, pathway_results, path_from_class, path_from_superclass, n_path, fp1, fp2 = classify_structure(smiles_string)

    output_list = []

    for result in classification_dict["pathway_results"]:
        output_dict = {}
        output_dict["type"] = "pathway"
        output_dict["entry"] = result
        output_list.append(output_dict)


    for result in classification_dict["superclass_results"]:
        output_dict = {}
        output_dict["type"] = "superclass"
        output_dict["entry"] = result
        output_list.append(output_dict)


    for result in classification_dict["class_results"]:
        output_dict = {}
        output_dict["type"] = "class"
        output_dict["entry"] = result
        output_list.append(output_dict)

    if classification_dict["isglycoside"]:
        output_dict = {}
        output_dict["type"] = "is glycoside"
        output_dict["entry"] = "glycoside"
        output_list.append(output_dict)

    #Creating Table
    white_list_columns = ["type", "entry"]
    table_fig = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in white_list_columns
        ],
        data=output_list,
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    )

    # Creating Structure Image for local
    molecule = Chem.MolFromSmiles(smiles_string)
    fig = Draw.MolToImage(molecule)
    img_obj = html.Img(src=fig, id='image')

    return [table_fig, img_obj]


def classify_structure(smiles):
    isglycoside = fingerprint_handler._isglycoside(smiles)

    fp = fingerprint_handler.calculate_fingerprint(smiles, 2)

    fp1 = np.asarray(fp[0].tolist()[0]).astype('float32')
    fp2 = np.asarray(fp[1].tolist()[0]).astype('float32')

    # Reshape for LOCAL model
    fp1 = fp1.reshape(1, -1)
    fp2 = fp2.reshape(1, -1)

    query_dict = {}
    query_dict["input_2048"] = fp1
    query_dict["input_4096"] = fp2

    # Handling SUPERCLASS
    # fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/SUPERCLASS:predict"
    # payload = json.dumps({"instances": [ query_dict ]})

    # headers = {"content-type": "application/json"}
    # json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    pred_super = np.asarray(super_model.predict([fp1,fp2])) # LOCAL prediction 
    n_super = list(np.where(pred_super[0]>=0.3)[0])

    path_from_superclass = []
    for j in n_super:
        path_from_superclass += ontology_dictionary['Super_hierarchy'][str(j)]['Pathway']
    path_from_superclass = list(set(path_from_superclass))

    query_dict = {}
    query_dict["input_2048"] = fp1
    query_dict["input_4096"] = fp2

    # Handling CLASS
    # fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/CLASS:predict"
    # payload = json.dumps({"instances": [ query_dict ]})

    # headers = {"content-type": "application/json"}
    # json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    pred_class = np.asarray(class_model.predict([fp1,fp2])) # LOCAL prediction 
    n_class = list(np.where(pred_class[0]>=0.1)[0])

    path_from_class = []
    for j in n_class:
        path_from_class += ontology_dictionary['Class_hierarchy'][str(j)]['Pathway']
    path_from_class = list(set(path_from_class))

    query_dict = {}
    query_dict["input_2048"] = fp1
    query_dict["input_4096"] = fp2

    # Handling PATHWAY
    # fp_pred_url = "http://npclassifier-tf-server:8501/v1/models/PATHWAY:predict"
    # payload = json.dumps({"instances": [ query_dict ]})

    # headers = {"content-type": "application/json"}
    # json_response = requests.post(fp_pred_url, data=payload, headers=headers)

    pred_path = np.asarray(pathway_model.predict([fp1,fp2])) # LOCAL prediction 
    n_path = list(np.where(pred_path[0]>=0.5)[0])

    class_result = []
    superclass_result = []
    pathway_result = []

    # Voting on Answer
    pathway_result, superclass_result, class_result, isglycoside = prediction_voting.vote_classification(n_path, 
                                                                                                        n_class, 
                                                                                                        n_super, 
                                                                                                        pred_class,
                                                                                                        pred_super, 
                                                                                                        path_from_class, 
                                                                                                        path_from_superclass, 
                                                                                                        isglycoside, 
                                                                                                        ontology_dictionary)

    
    return isglycoside, class_result, superclass_result, pathway_result, path_from_class, path_from_superclass, n_path, fp1, fp2


def _process_full_classification(smiles_string):
    isglycoside, class_results, superclass_results, pathway_results, path_from_class, path_from_superclass, n_path, fp1, fp2 = classify_structure(smiles_string)

    respond_dict = {}
    respond_dict["class_results"] = class_results
    respond_dict["superclass_results"] = superclass_results
    respond_dict["pathway_results"] = pathway_results
    respond_dict["isglycoside"] = isglycoside
    
    respond_dict["fp1"] = fp1
    respond_dict["fp2"] = fp2
    
    return respond_dict


@server.route("/classify")
def classify():
    smiles_string = request.values.get("smiles")
    respond_dict = _process_full_classification(smiles_string)

    return json.dumps(respond_dict)


if __name__ == "__main__":
    app.run_server(debug=True, port=8080)

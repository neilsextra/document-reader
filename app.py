from flask import Flask, Blueprint, render_template, request, send_file, Response
from flask_bower import Bower
import io
from os import environ
import datetime
import string
import json
import sys
import azure
import os
import requests
import time
import re
import operator
import numpy as np
from requests import get, post
from pathlib import Path

from PyPDF2 import PdfFileReader, PdfFileWriter

from datetime import datetime, timedelta

from azure.storage.filedatalake import DataLakeServiceClient
from azure.core._match_conditions import MatchConditions
from azure.storage.filedatalake._models import ContentSettings

views = Blueprint('views', __name__, template_folder='templates')

app = Flask(__name__)

Bower(app)

app.register_blueprint(views)

my_content_settings = ContentSettings(content_type='application/pdf')

def log(f, message):
    f.write("%s : %s\n" % (str(datetime.now()), message))
    f.flush()


def get_configuration():
    debug_file = "debug.log"
    max_tries = "15"    
    wait_sec = "5"
    max_wait_sec = "60"
    terms = {}

    try:
        import configuration as config
        
        debug_file = config.DEBUG_FILE
        max_tries = config.MAX_TRIES
        wait_sec = config.WAIT_SEC
        max_wait_sec = config.MAX_WAIT_SEC
        terms = config.TERMS
    except ImportError:
        pass

    debug_file = environ.get('DEBUG_FILE', debug_file)
    max_tries = environ.get('MAX_TRIES', max_tries)
    wait_sec = environ.get('WAIT_SEC', max_tries)
    max_wait_sec = environ.get('WAIT_SEC', max_wait_sec)
    terms = environ.get('TERMS', terms)

    return {
        'debug_file': debug_file,
        'max_tries': max_tries,
        'wait_sec': wait_sec,
        'max_wait_sec': max_wait_sec,
        'terms': terms
    }


def get_file_system_client(account, token, container):

    service_client = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
        "https", account), credential=token)

    return service_client.get_file_system_client(container)


def write_data_to_file(container_client, container_directory, uploaded_file, data, data_length):
    directory_client = container_client.get_directory_client(
        container_directory)

    file_client = directory_client.get_file_client(uploaded_file)

    file_client.create_file()

    file_client.append_data(data,  offset=0, length=data_length)
    print("data_offset: %d" % (data_length))
    file_client.flush_data(data_length)

    print("Written Data To File")


def train_model(f, form_url, apim_key,  account, token, container, directory):
    configuration = get_configuration()

    post_url = form_url + r"/formrecognizer/v2.0-preview/custom/models"

    source = "https://%s.blob.core.windows.net/%s%s" % (account, container, token)

    if directory == "/":
        source = "https://%s.blob.core.windows.net/%s%s" % (account, container, token)
    else:
        source = "https://%s.blob.core.windows.net/%s%s%s" % (account, container, directory, token)
      
    # Path to the folder in blob storage where your forms are located. If your forms are at the root of your container, leave this string empty.
    prefix = ""
    includeSubFolders = True
    useLabelFile = False

    headers = {
        # Request headers
        'Content-Type': 'application/pdf',
        'Ocp-Apim-Subscription-Key': apim_key,
    }

    body = {
        "source": source,
        "sourceFilter": {
            "prefix": prefix,
            "includeSubFolders": includeSubFolders
        },
        "useLabelFile": useLabelFile
    }

    print(json.dumps(body, sort_keys=True))

    try:
        resp = post(url=post_url, json=body, headers=headers)
        if resp.status_code != 201:
            log(f, "[Train] POST model failed (%s):\n%s" %
                  (resp.status_code, json.dumps(resp.json())))
        print("POST model succeeded:\n%s" % resp.headers)
        get_url = resp.headers["location"]

    except Exception as e:
        print("POST model failed:\n%s" % str(e))
        return False

    n_tries = int(configuration['max_tries'])
    n_try = 0
    wait_sec = int(configuration['wait_sec'])
    max_wait_sec = int(configuration['max_wait_sec'])

    while n_try < n_tries:
        print("tries %d" % n_try)

        try:
            resp = get(url=get_url, headers=headers)
            resp_json = resp.json()
            if resp.status_code != 200:
                print("GET model failed (%s):\n%s" %
                      (resp.status_code, json.dumps(resp_json)))
                return {
                    'status': 'fail',
                    'status_code' : resp.status_code,
                    'message' : json.dumps(resp_json)

                }
            
            model_status = resp_json["modelInfo"]["status"]       

            print("model_status %s" % model_status)

            if model_status == "ready":
                print("\n%s" % json.dumps(resp_json, indent=4, sort_keys=True))
                modelID = resp_json["modelInfo"]["modelId"]
                
                training_documents = resp_json["trainResult"]["trainingDocuments"]

                log(f, "[Training] suceeded : ModelID:\n%s" % modelID)

                return modelID, training_documents

            if model_status == "invalid":
                print("Training failed. Model is invalid:\n%s" %
                      json.dumps(resp_json))
            # Training still running. Wait and retry.
            time.sleep(wait_sec)
            n_try += 1
            wait_sec = min(2*wait_sec, max_wait_sec)
        except Exception as e:
            msg = "GET model failed:\n%s" % str(e)
            print(msg)
            return {
                'status': 'fail',
                'message' : str(e)
            }

    return (resp_json)

def analyze_form(f, post_url, apim_key, modelID, data):
    configuration = get_configuration()

    # Endpoint URL
    url = post_url + "/formrecognizer/v2.0-preview/custom/models/%s/analyze" % modelID

    params = {
        "includeTextDetails": True
    }

    headers = {
        # Request headers
        'Content-Type': 'application/pdf',
        'Ocp-Apim-Subscription-Key': apim_key,
    }

    try:
        resp = post(url=url, data=data, headers=headers, params=params)
        if resp.status_code != 202:
            print("POST analyze failed:\n%s" % json.dumps(resp.json()))

        print("POST analyze succeeded:\n%s" % resp.headers)
        get_url = resp.headers["operation-location"]

    except Exception as e:
        print("POST analyze failed:\n%s" % str(e))

    n_tries = int(configuration['max_tries'])
    n_try = 0
    wait_sec = int(configuration['wait_sec'])
    max_wait_sec = int(configuration['max_wait_sec'])

    while n_try < n_tries:
        try:
            resp = get(url=get_url, headers={
                       "Ocp-Apim-Subscription-Key": apim_key})
            resp_json = resp.json()
            if resp.status_code != 200:
                print("GET analyze results failed:\n%s" %
                      json.dumps(resp_json))
            status = resp_json["status"]
            if status == "succeeded":
                print("Analysis succeeded")
  
                result = analyze_response(resp_json)

                return result
            if status == "failed":
                print("Analysis failed:\n%s" % json.dumps(resp_json))
            # Analysis still running. Wait and retry.
            time.sleep(wait_sec)
            n_try += 1
            wait_sec = min(2*wait_sec, max_wait_sec)
        except Exception as e:
            log(f, "[Analyze] GET analyze results failed: %s" % str(e))
            return {
                'status' : 'fail',
                'message' : str(e)
            }


def analyze_text(page_no, text):
    configuration = get_configuration()
   
    keys = list(configuration['terms'].keys())

    for key in keys:

        if configuration['terms'][key]['text'] == text and int(configuration['terms'][key]['page']) == page_no:

            print("%s==%s" % (configuration['terms'][key]['text'], text))
            
            configuration['terms'][key]

            return configuration['terms'][key]

    return None        

            

def analyze_response(resp_json):
    configuration = get_configuration()

    page_results = resp_json['analyzeResult']['pageResults']

    page_no = 0

    items = []

    for page_result in page_results:

        key_value_pairs= page_result['keyValuePairs']
        page_no += 1

        for key_value_pair in key_value_pairs:
            key = key_value_pair['key']
            value = key_value_pair['value']
            
            entry = analyze_text(page_no, key['text'])

            if entry != None:
                item = {}
                item['entry'] = entry    
                item['result'] = {
                    'key' : {
                        'text' : key['text'],
                        'bounding_box' : key['boundingBox']
                    },
                    'value' : {
                        'text' : value['text'],
                        'bounding_box' : value['boundingBox'],
                       
                    }
                }

                print(json.dumps(item))

                items.append(item)

    return items


@ app.route("/query", methods=["GET"])
def query():
    configuration = get_configuration()

    f = open(configuration['debug_file'], 'a')

    output = []

    try:
        account = request.values.get('account')  # the Datalake Account
        token = request.values.get('token')  # The Datalake SAF token
        container = request.values.get('container')  # the Datalake Container
        directory = request.values.get('directory')  # the Datalake Directory

        print("%s : %s : %s : %s" % (account, token, container, directory))

        file_system_client = get_file_system_client(account, token, container)
        directory_client = file_system_client.get_directory_client(directory)

        directory_properties = directory_client.get_directory_properties()
        print(directory_properties)

        directory_metadata = {}
        
        if 'modelId' in directory_properties['metadata']:
            directory_metadata = directory_properties['metadata']

        paths = []
   
        path_list = file_system_client.get_paths(
            path=directory, recursive=False)

        for path in path_list:
            file_client = directory_client.get_file_client(path)

            file_properties = file_client.get_file_properties()
            
            if 'modelId' in file_properties['metadata']:
                paths.append({
                    "path": path.name,
                    "is_directory": path.is_directory,
                    "modelId": file_properties['metadata']['modelId'],
                    "pages": file_properties['metadata']['pages']           
                })            
            else:
                paths.append({
                    "path": path.name,
                    "is_directory": path.is_directory
                })

        output.append({
            "directory_metadata" : directory_metadata,
            "paths": paths,
            "status": "OK"
        })

        return json.dumps(output, sort_keys=True), 200

    except Exception as e:

        log(f, str(e))
        f.close()

        output.append({
            "status": 'fail',
            "error": str(e)
        })

        return json.dumps(output, sort_keys=True), 500  


@app.route("/upload", methods=["POST"])
def upload():
    configuration = get_configuration()

    f = open(configuration['debug_file'], 'a')
    
    cloud_account = request.values.get('cloud_account')
    cloud_token = request.values.get('cloud_token')
    cloud_container = request.values.get('cloud_container')
    cloud_directory = request.values.get('cloud_directory')
    file_size = int(request.values.get('file_size'))

    log(f, '[UPLOAD] commenced uploading - %s:%s:%s - %d' %(cloud_account, cloud_container, cloud_directory, file_size))

    output = []

    try:
        log(f, '[UPLOAD] Read File Content')

        uploaded_files = request.files

        log(f, '[UPLOAD] Files %d' % (len(uploaded_files)))

        files = []

        for uploaded_file in uploaded_files:
            fileContent = request.files.get(uploaded_file)

            file_system_client = get_file_system_client(cloud_account, cloud_token, cloud_container)
 
            write_data_to_file(file_system_client, cloud_directory, uploaded_file, fileContent, file_size)

            log(f, "[UPLOAD] Uploaded  file '%s'" % uploaded_file)


        output.append({
            "filenames": files,
            "status": "OK"
        })

        log(f, '[UPLOAD] Completed Upload')

        f.close()

        return json.dumps(output, sort_keys=True), 200

    except Exception as e:

        print(str(e))

        log(f, str(e))
        f.close()

        output.append({
            "status": 'fail',
            "error": str(e)
        })

        return json.dumps(output, sort_keys=True), 500

@app.route("/retrieve", methods=["GET"])
def retrieve():
    account = request.values.get('account')
    token = request.values.get('token')
    container = request.values.get('container')
    directory = request.values.get('directory')
    filename = request.values.get('filename')

    file_system_client = get_file_system_client(account, token, container)

    directory_client = file_system_client.get_directory_client(directory)

    file_client = directory_client.get_file_client(filename)

    download_stream = file_client.download_file()

    return Response(io.BytesIO(download_stream.readall()), mimetype='application/pdf')


@app.route("/train", methods=["GET"])
def train():
    configuration = get_configuration()

    account = request.values.get('account')
    token = request.values.get('token')
    container = request.values.get('container')
    directory = request.values.get('directory')
    form_url = request.values.get('formURL')
    apim_key = request.values.get('apimKey')

    f = open(configuration['debug_file'], 'a')

    log(f, '[TRAIN] training commenced')
     
    modelId, training_documents = train_model(f, form_url, apim_key, account, token, container, directory)
     
    file_system_client = get_file_system_client(account, token, container)
    directory_client = file_system_client.get_directory_client(directory)
    directory_client.set_metadata(
        {'modelId': modelId,
          'forms_recognizer_url' : form_url,
          'apim_key' : apim_key
        }      
    )

    for training_document in training_documents:
        log(f, '[TRAIN] trained %s' % training_document['documentName'])
        file_client = directory_client.get_file_client(training_document['documentName'])
        file_client.set_metadata({'modelId': modelId,
                                  'forms_recognizer_url' : form_url,
                                  'apim_key' : apim_key,
                                  'pages': str(training_document['pages'])})

    log(f, '[TRAIN] training finished')
    f.close()
  
    output = {
        'ModelId' : modelId
    }
 
    return json.dumps(output, sort_keys=True), 200

@app.route("/analyze", methods=["GET"])
def analyze():
    configuration = get_configuration()
    
    f = open(configuration['debug_file'], 'a')

    account = request.values.get('account')
    token = request.values.get('token')
    container = request.values.get('container')
    directory = request.values.get('directory')
    filename = request.values.get('filename')

    file_system_client = get_file_system_client(account, token, container)

    directory_client = file_system_client.get_directory_client(directory)
    directory_properties = directory_client.get_directory_properties()
    print(directory_properties)
        
    if 'modelId' in directory_properties['metadata']:
    
        file_client = directory_client.get_file_client(filename)

        download_stream = file_client.download_file()

        
        result = analyze_form(f, directory_properties['metadata']['forms_recognizer_url'], 
                              directory_properties['metadata']['apim_key'], 
                              directory_properties['metadata']['modelId'], 
                              io.BytesIO(download_stream.readall()))
        
        f.close()

        return json.dumps(result, sort_keys=True), 200

    else:
        output = {
            'Status' : "fail",
            'Message' : 'No model found'
        }
        
        return json.dumps(output, sort_keys=True), 500


@app.route("/")
def start():
    return render_template("main.html")


if __name__ == "__main__":
    PORT = int(environ.get('PORT', '8080'))
    app.run(host='0.0.0.0', port=PORT)
#!/bin/bash

PATH_TO_NOTEBOOK='notebooks/5 - Deploy_model.ipynb'
PATH_TO_YAML='deploy_params.yaml'
BASE64_ENCODED_PARAMS=$(cat $PATH_TO_YAML | base64)

echo "PATH_TO_NOTEBOOK: $PATH_TO_NOTEBOOK"
echo "PATH_TO_YAML: $PATH_TO_YAML"
echo "BASE64_ENCODED_PARAMS: $BASE64_ENCODED_PARAMS"

curl --location --request POST 'https://wps-ehtdmb-85138fe6-e211-4b8f-8371-70f18d002465.hub.eox.at/wps' \
--header 'Content-Type: application/xml' \
--data-raw '<?xml version="1.0"?>
<wps:Execute xmlns:wps="http://www.opengis.net/wps/2.0" xmlns:ows="http://www.opengis.net/ows/2.0" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" service="WPS" version="2.0.0" xsi:schemaLocation="http://www.opengis.net/wps/2.0 http://schemas.opengis.net/wps/2.0/wpsExecute.xsd" response="document" mode="async">
    <ows:Identifier>execute-notebook</ows:Identifier>
    <wps:Input id="Notebook">
        <wps:Data>
            <wps:LiteralValue>$PATH_TO_NOTEBOOK</wps:LiteralValue>
        </wps:Data>
    </wps:Input>
    <wps:Input id="Parameters">
        <wps:Data>
            <wps:LiteralValue>$BASE64_ENCODED_PARAMS</wps:LiteralValue>
        </wps:Data>
    </wps:Input>
    <wps:Output id="Result" transmission="reference"/>
</wps:Execute>'

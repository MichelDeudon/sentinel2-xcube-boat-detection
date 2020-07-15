#!/bin/bash

read  -p 'Job id: ' JID
echo "Query: job id: $JID"

curl --location --request POST 'https://wps-ehtdmb-85138fe6-e211-4b8f-8371-70f18d002465.hub.eox.at/wps' \
--header 'Content-Type: application/xml' \
--data-raw '<wps:GetStatus xmlns:wps="http://www.opengis.net/wps/2.0" xmlns:ows="http://www.opengis.net/ows/2.0" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.opengis.net/wps/2.0 ../wps.xsd" service="WPS" version="2.0.0">
        <wps:JobID>'$JID'</wps:JobID>
</wps:GetStatus>'
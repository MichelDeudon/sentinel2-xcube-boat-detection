# ☄️ sentinel2-xcube-boat-detection

Github repository to detect and counts boat traffic 🛥️ in [Sentinel-2 imagery](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) using temporal, spectral and spatial features.

## ⛵ Setup

##### With EDC
- Clone this repository in a Jupyter Lab environment (Python 3.6+) on Euro Data Cube Dashboard (requires a valid account)
- Edit Sentinel Hub credentials and [Mapbox](https://studio.mapbox.com/) token in a .env file (requires a valid account).
- Run ```pip install -r requirements.txt```
- TODO: Edit Docker image.

## 🛰️ Pipeline

### 1. 📷 Annotate 1 squared km chips with boat counts.

Download [Sentinel 2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) L1C products (bands B03, B08, [CLP](https://github.com/sentinel-hub/sentinel2-cloud-detector)) from [Sentinel Hub](https://www.sentinel-hub.com/) via [xcube-sh](https://github.com/dcs4cop/xcube-sh). Remove nans and cloudy images with CLP. Compute a background NDWI image by fusion over time (max).

### 2. 🔭 Learn to detect and count boat traffic

- Input (1, 2 or 3 channels): NIR, Background NDWI and CLP.
- Model: Embed/encode pixels, embed/encode patches, predict presence/counts.

### 3. 🗺️ Deploy

Deploy model on large AOI (Ports, Straits, MPA), e.g. Gibraltar Strait (196 squared km), by running Notebook 3 and editing data/aoi.json.

## Links

###### 📡 2020/06/05 ESA [RACE Dashboard](https://race.esa.int/)
###### 📡 2020/05/06 CNES [SpaceGate Article](https://spacegate.cnes.fr/fr/covid-19-venise-sans-les-bateaux)
###### 🛰️ 2020/04/15 ESA [Tweet](https://mobile.twitter.com/EO_OPEN_SCIENCE/status/1250367319936765953)
###### 🛰️ 2020/04/06 ESA [Covid-19 Custom Script Contest](https://www.sentinel-hub.com/contest)

## Credits

[ESA](https://www.esa.int/), [Copernicus](https://scihub.copernicus.eu/dhus/#/home), [Euro Data Cube](https://eurodatacube.com/), [Sinergise](https://www.sinergise.com/)

## License

This repository is made available under the [Do No Harm License](https://github.com/raisely/NoHarm). <br>
Copyright (c) 2020 Michel Deudon and Zhichao Lin. All rights reserved. 

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />Our dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.


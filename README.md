# Salzburg Forest Cover Analysis App

This is a web-based GIS application for analysing forest cover changes in the Salzburg region of Austria from 2018 to 2024 using ESRI Land Cover Explorer data.

Initial opening view:
<img width="3071" height="1637" alt="Screenshot 2025-08-01 110729" src="https://github.com/user-attachments/assets/2925d523-f015-40a2-a767-59c5617bcef3" />
Map view:
<img width="3071" height="1638" alt="Screenshot 2025-08-01 110749" src="https://github.com/user-attachments/assets/53378029-6024-4c82-9dd1-1c2acb388ed7" />

## Overview

This application examines temporal changes in forest cover across the Salzburg federal state by leveraging freely available data from ESRI's Land Cover Explorer. The analysis focuses on change in tree cover over a 6-year study period to provide insights into environmental changes in the region.

## Data Sources

- **Land Cover Data**: ESRI Land Cover Explorer - Freely available global land cover data
- **Administrative Boundaries**: GADM - Salzburg federal state boundaries at Administrative Level 2
- **Temporal Range**: 2018-2024
- **Geographic Focus**: Salzburg, Austria

## Project Structure

```
gisapp/
├── asset/                      # Contains header images
├── salzburg_raster/           # Raster data files for the Salzburg region
├── admin_shp_downl.py         # Downloads GADM administrative boundary data
├── all_data_extract.py        # Main data extraction and processing script
├── env.yaml                   # Conda/Python environment configuration
├── index.html             # Main web application interface
├── main.js                    # JavaScript functionality for web app
├── salzburg_AUT2.geojson      # Salzburg administrative boundaries (GeoJSON)
├── salzburg_enhanced.geojson  # Enhanced Salzburg boundaries with additional data created in 'all_data_extract.py'
├── style.css                  # Web application styling
└── README.md                  # This file
```

## File Descriptions

### Core Application Files

- **`index.html`**: Main web application interface. Provides the user interface for visualising forest cover changes and interacting with the geospatial data.

- **`main.js`**: JavaScript functionality powering the web application. Handles map interactions, data visualisation, and user interface dynamics.

- **`style.css`**: Defines the visual styling and layout of the interface.

### Data Processing Scripts

- **`admin_shp_downl.py`**: Python script for downloading Salzburg administrative boundaries from the GADM database. Converts shapefile data for use in the application.

- **`all_data_extract.py`**: Main data extraction and processing script. Handles processing of ESRI Land Cover Explorer data across multiple years.

### Data Files

- **`salzburg_AUT2.geojson`**: GeoJSON file containing Salzburg federal state administrative boundaries at Level 2 resolution. Used as the base geographic layer for analysis.

- **`salzburg_enhanced.geojson`**: Enhanced version of the Salzburg boundaries with additional attributes and processed data for improved analysis and visualisation.

### Supporting Files

- **`salzburg_raster/`**: Directory containing raster data files for the Salzburg region from ESRI Land Cover Explorer.

- **`asset/`**: Contains header images for use in the web application interface.

- **`env.yaml`**: Conda environment configuration file specifying Python dependencies and package versions required for the data processing scripts.

ESRI LULC data was manually downloaded from [here](https://livingatlas.arcgis.com/landcoverexplorer/).
This required selecting the desired year from the timeline (2018-2024) and using the download tools to export land cover data for the region.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/placcky/gisapp.git
   cd gisapp
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f env.yaml
   conda activate gisapp
   ```

3. **Run data processing scripts:**
   ```bash
   # Download administrative boundaries
   python admin_shp_downl.py
   
   # Extract and process land cover data
   python all_data_extract.py
   ```

## Usage

### Web Application

1. Open `indexrev3.html` in a modern web browser
2. The application will load the Salzburg region map with forest cover data
3. Use the interface controls to explore different years and visualise changes
4. Interactive features are powered by `main.js` and styled with `style.css`

### Data Processing

- **Administrative Boundaries**: Run `admin_shp_downl.py` to download and process Salzburg boundary data from GADM
- **Land Cover Data**: Execute `all_data_extract.py` to fetch and process ESRI Land Cover Explorer data for multiple years

## Technical Stack

- **Backend**: Python with geospatial libraries
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Formats**: GeoJSON, Raster data
- **Mapping**: Web-based GIS visualisation

## Key Features

- Interactive web-based map interface
- Temporal analysis of forest cover changes (2018-2024)
- Administrative boundary integration
- Raster data processing and visualisation
- Responsive web design


## Data Processing Workflow

1. **Boundary Data**: `admin_shp_downl.py` downloads Salzburg administrative boundaries
2. **Land Cover Data**: `all_data_extract.py` processes ESRI land cover data
3. **Web Visualisation**: `indexrev3.html`, `main.js`, and `style.css` provide an interactive interface
4. **Enhanced Data**: `salzburg_enhanced.geojson` contains processed results



*This project analyses environmental changes in the Salzburg region to support conservation efforts and land use planning decisions.*

import geopandas as gpd
import os
import urllib.request
import zipfile

# GADM v4.1 ZIP file URL - Austria, admin level 2
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_AUT_shp.zip"
zip_path = "gadm41_AUT_shp.zip"
extract_dir = "gadm_aut"

# Downloading, if it's not yet downloaded
if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)

# Unzip
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Shapefile reading
gdf = gpd.read_file(os.path.join(extract_dir, "gadm41_AUT_2.shp"))

# Plot
gdf.plot(edgecolor="black", cmap="Pastel1")


# Second code to RUN
# This code converts the specified and selected administrative level boundary .shp to .geojson for later use

import geopandas as gpd
import os

base_dir = os.path.dirname(__file__)
shp_path = os.path.join(base_dir, "gadm_aut", "gadm41_AUT_2.shp")

# Beolvasás
gdf = gpd.read_file(shp_path)

# Nézd meg az oszlopokat, hogy biztosan legyen 'NAME_1' (ez a tartomány neve)
print(gdf.columns)
print(gdf['NAME_1'].unique())  # Milyen tartománynevek vannak?

# Szűrés Salzburgra
salzburg_gdf = gdf[gdf['NAME_1'] == "Salzburg"]

# Mentés
geojson_path = os.path.join(base_dir, "salzburg_AUT2.geojson")
salzburg_gdf.to_file(geojson_path, driver="GeoJSON")

print("Successfully saved Salzburg region to GeoJSON:", geojson_path)

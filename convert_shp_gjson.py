import geopandas as gpd

# SHP file reading
shp_path = r"C:\Users\emese\Documents\ErasmusGIS\semester_2\ApplicationDevelopmentGIS\final_project\salzburg_data\salzburg_AUT1.shp"  # vagy az elérési útvonalad
gdf = gpd.read_file(shp_path)

# GeoJSON file saving
geojson_path = "salzburg_AUT1.geojson"
gdf.to_file(geojson_path, driver="GeoJSON")

print("Successfully saved to GeoJSON:", geojson_path)

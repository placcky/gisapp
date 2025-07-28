import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.mask import mask
import os
import re

def calculate_land_cover_analysis(tiff_files, geojson_file):
    """
    Calculate land cover analysis including area calculations and class 2 specific metrics.
    
    Parameters:
    tiff_files (list): List of paths to TIFF files
    geojson_file (str): Path to GeoJSON file containing AOIs
    
    Returns:
    tuple: (all_classes_df, class2_summary_df, detailed_df, aoi_attributes)
    """
    
    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_file)
    print(f"Loaded {len(gdf)} AOIs from GeoJSON")
    
    # Ensure GeoJSON is in a projected CRS for area calculations
    if gdf.crs.is_geographic:
        # Convert to appropriate projected CRS (UTM zone for Austria)
        gdf = gdf.to_crs('EPSG:31287')  # MGI / Austria GK East
        print("Converted GeoJSON to projected CRS for area calculations")
    
    # Calculate polygon areas in km²
    gdf['area_km2'] = gdf.geometry.area / 1_000_000
    
    results = {}
    all_classes = set()
    
    # Process each TIFF file
    for tiff_file in tiff_files:
        print(f"\nProcessing: {tiff_file}")
        year = extract_year_from_filename(tiff_file)
        
        with rasterio.open(tiff_file) as src:
            print(f"  TIFF CRS: {src.crs}, Shape: {src.shape}")
            
            # Calculate pixel area in km²
            pixel_area_km2 = abs(src.transform[0] * src.transform[4]) / 1_000_000
            
            # Reproject GeoJSON to match TIFF CRS if needed
            if gdf.crs != src.crs:
                print(f"  Reprojecting GeoJSON from {gdf.crs} to {src.crs}")
                gdf_reproj = gdf.to_crs(src.crs)
                # Recalculate areas in the new CRS
                gdf_reproj['area_km2'] = gdf_reproj.geometry.area / 1_000_000
            else:
                gdf_reproj = gdf.copy()
            
            aoi_results = []
            
            # Process each AOI
            for idx, row in gdf_reproj.iterrows():
                aoi_name = row.get('NAME_2', f'AOI_{idx}')
                geometry = [row.geometry.__geo_interface__]
                polygon_area_km2 = row['area_km2']
                
                try:
                    # Mask the raster with the AOI geometry
                    masked_data, masked_transform = mask(
                        src, geometry, crop=True, nodata=src.nodata
                    )
                    
                    band_data = masked_data[0]
                    
                    # Remove nodata values
                    if src.nodata is not None:
                        valid_data = band_data[band_data != src.nodata]
                    else:
                        valid_data = band_data[~np.isnan(band_data)]
                    
                    if len(valid_data) == 0:
                        print(f"    {aoi_name}: No valid data")
                        class_percentages = {}
                        class_counts = {}
                        total_pixels = 0
                        class2_area_km2 = 0
                        class2_percentage = 0
                    else:
                        # Get unique classes and their counts
                        unique_classes, counts = np.unique(valid_data, return_counts=True)
                        total_pixels = len(valid_data)
                        all_classes.update(unique_classes)
                        
                        # Calculate class 2 specific metrics only
                        class2_count = 0
                        for class_val, count in zip(unique_classes, counts):
                            if int(class_val) == 2:
                                class2_count = count
                                break
                        
                        class2_area_km2 = class2_count * pixel_area_km2
                        class2_percentage = (class2_count / total_pixels) * 100 if total_pixels > 0 else 0
                        
                        print(f"    {aoi_name}: {len(unique_classes)} classes, Class 2: {class2_percentage:.2f}% ({class2_area_km2:.3f} km²)")
                    
                    # Store results
                    aoi_result = {
                        'AOI_Name': aoi_name,
                        'AOI_ID': row.get('GID_2', f'ID_{idx}'),
                        'Polygon_Area_km2': polygon_area_km2,
                        'Total_Pixels': total_pixels,
                        'Class2_Area_km2': class2_area_km2,
                        'Class2_Percentage': class2_percentage,
                        'Class2_Count': class2_count
                    }
                    aoi_results.append(aoi_result)
                    
                except Exception as e:
                    print(f"    Error processing {aoi_name}: {str(e)}")
                    aoi_results.append({
                        'AOI_Name': aoi_name,
                        'AOI_ID': row.get('GID_2', f'ID_{idx}'),
                        'Polygon_Area_km2': row['area_km2'],
                        'Total_Pixels': 0,
                        'Class2_Area_km2': 0,
                        'Class2_Percentage': 0,
                        'Class2_Count': 0
                    })
            
            results[year] = aoi_results
    
    # Create DataFrames
    all_classes_df, class2_summary_df, detailed_df, aoi_attributes = create_results_dataframes(results, all_classes)
    return all_classes_df, class2_summary_df, detailed_df, aoi_attributes

def extract_year_from_filename(filename):
    """Extract year from filename."""
    # Look for 4-digit year pattern (2018-2024 based on your files)
    year_match = re.search(r'(20[12][0-9])', os.path.basename(filename))
    if year_match:
        return int(year_match.group(1))
    else:
        # Try to extract from filename like "18.tif" -> 2018
        number_match = re.search(r'(\d{2})\.tif', os.path.basename(filename))
        if number_match:
            year_num = int(number_match.group(1))
            return 2000 + year_num if year_num >= 18 else 2000 + year_num
        return os.path.basename(filename)

def create_results_dataframes(results, all_classes):
    """Create DataFrames from results dictionary - focused on Class 2 only."""
    
    # Create detailed DataFrame
    detailed_data = []
    class2_summary_data = {}
    aoi_attributes = {}  # Store attributes for GeoJSON
    
    for year, aoi_list in results.items():
        for aoi_data in aoi_list:
            aoi_name = aoi_data['AOI_Name']
            
            # Store AOI attributes for GeoJSON (only once per AOI)
            if aoi_name not in aoi_attributes:
                aoi_attributes[aoi_name] = {
                    'AOI_ID': aoi_data['AOI_ID'],
                    'Polygon_Area_km2': aoi_data['Polygon_Area_km2']
                }
            
            # Add yearly Class 2 data to AOI attributes
            aoi_attributes[aoi_name][f'{year}_Class2_Area_km2'] = round(aoi_data['Class2_Area_km2'], 3)
            aoi_attributes[aoi_name][f'{year}_Class2_Perc'] = round(aoi_data['Class2_Percentage'], 2)
            aoi_attributes[aoi_name][f'{year}_Class2_Count'] = aoi_data['Class2_Count']
            
            # Add to class 2 summary
            class2_summary_data[aoi_name] = class2_summary_data.get(aoi_name, {})
            class2_summary_data[aoi_name][year] = {
                'Area_km2': aoi_data['Class2_Area_km2'],
                'Percentage': aoi_data['Class2_Percentage'],
                'Count': aoi_data['Class2_Count'],
                'Polygon_Area_km2': aoi_data['Polygon_Area_km2']
            }
            
            # Add detailed data for Class 2 only
            detailed_data.append({
                'Year': year,
                'AOI_Name': aoi_name,
                'AOI_ID': aoi_data['AOI_ID'],
                'Polygon_Area_km2': aoi_data['Polygon_Area_km2'],
                'Class2_Percentage': aoi_data['Class2_Percentage'],
                'Class2_Area_km2': aoi_data['Class2_Area_km2'],
                'Class2_Count': aoi_data['Class2_Count'],
                'Total_Pixels': aoi_data['Total_Pixels']
            })
    
    # Calculate summary statistics for each AOI
    for aoi_name, attrs in aoi_attributes.items():
        # Class 2 statistics
        class2_areas = [v for k, v in attrs.items() if k.endswith('_Class2_Area_km2')]
        class2_percs = [v for k, v in attrs.items() if k.endswith('_Class2_Perc')]
        
        if class2_areas:
            attrs['Class2_Avg_Area_km2'] = round(np.mean(class2_areas), 3)
            attrs['Class2_Avg_Perc'] = round(np.mean(class2_percs), 2)
            attrs['Class2_Max_Area_km2'] = round(max(class2_areas), 3)
            attrs['Class2_Min_Area_km2'] = round(min(class2_areas), 3)
            attrs['Class2_Total_Change'] = round(class2_areas[-1] - class2_areas[0], 3)
    
    # Create DataFrames
    df_detailed = pd.DataFrame(detailed_data)
    
    # Create Class 2 pivot table (AOIs as rows, years as columns)
    all_classes_df = df_detailed.pivot_table(
        index=['AOI_Name', 'Polygon_Area_km2'], 
        columns='Year', 
        values='Class2_Percentage', 
        fill_value=0
    ).round(2)
    
    # Create Class 2 specific summary
    class2_rows = []
    for aoi_name, year_data in class2_summary_data.items():
        row = {'AOI_Name': aoi_name}
        polygon_area = None
        
        for year, metrics in year_data.items():
            row[f'{year}_Area_km2'] = round(metrics['Area_km2'], 3)
            row[f'{year}_Percentage'] = round(metrics['Percentage'], 2)
            row[f'{year}_Count'] = metrics['Count']
            if polygon_area is None:
                polygon_area = metrics['Polygon_Area_km2']
        
        row['Polygon_Area_km2'] = round(polygon_area, 2)
        class2_rows.append(row)
    
    class2_summary_df = pd.DataFrame(class2_rows).set_index('AOI_Name')
    
    return all_classes_df, class2_summary_df, df_detailed, aoi_attributes

def create_enhanced_geojson(original_geojson_file, aoi_attributes, output_path):
    """
    Create an enhanced GeoJSON with calculated land cover statistics.
    
    Parameters:
    original_geojson_file (str): Path to original GeoJSON file
    aoi_attributes (dict): Dictionary with calculated attributes for each AOI
    output_path (str): Path for output enhanced GeoJSON
    """
    
    # Load original GeoJSON
    gdf = gpd.read_file(original_geojson_file)
    print(f"Creating enhanced GeoJSON with {len(gdf)} polygons...")
    
    # Add calculated attributes to each polygon
    for idx, row in gdf.iterrows():
        aoi_name = row.get('NAME_2', f'AOI_{idx}')
        
        if aoi_name in aoi_attributes:
            # Add all calculated attributes
            for attr_name, attr_value in aoi_attributes[aoi_name].items():
                gdf.at[idx, attr_name] = attr_value
        else:
            print(f"Warning: No calculated data found for {aoi_name}")
    
    # Save enhanced GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Enhanced GeoJSON saved: {output_path}")
    
    # Print attribute summary
    print(f"\nEnhanced GeoJSON includes:")
    print(f"  - Original attributes: {len([c for c in gdf.columns if not c.endswith(('_km2', '_Perc'))])}")
    print(f"  - New calculated attributes: {len([c for c in gdf.columns if c.endswith(('_km2', '_Perc'))])}")
    
    return gdf

def main():
    """Main execution function."""
    
    # Define file paths
    tiff_files = [
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\18.tif",
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\19.tif",
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\20.tif",
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\21.tif",
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\22.tif",
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\23.tif",
        r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_raster\24.tif"
    ]
    
    geojson_file = r"C:\Users\maria\Documents\PLUS\2_Modules\SS25\Spatial_analysis_R_python\Final_proj\gisapp-main\salzburg_AUT2.geojson"
    
    # Check if files exist
    missing_files = [f for f in tiff_files + [geojson_file] if not os.path.exists(f)]
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    # Run analysis
    print("Starting land cover analysis...")
    all_classes_df, class2_summary_df, detailed_df, aoi_attributes = calculate_land_cover_analysis(
        tiff_files, geojson_file
    )
    
    # Display results
    print("\n" + "="*60)
    print("CLASS 2 LAND COVER SUMMARY (Area and Percentage)")
    print("="*60)
    print(class2_summary_df)
    
    print("\n" + "="*60)
    print("ALL CLASSES PERCENTAGE SUMMARY")
    print("="*60)
    print(all_classes_df)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files
    class2_summary_df.to_csv(f"{output_dir}/class2_landcover_summary.csv")
    all_classes_df.to_csv(f"{output_dir}/all_classes_percentage_summary.csv")
    detailed_df.to_csv(f"{output_dir}/detailed_analysis.csv", index=False)
    
    # Create and save enhanced GeoJSON
    enhanced_geojson_path = f"{output_dir}/salzburg_enhanced.geojson"
    enhanced_gdf = create_enhanced_geojson(geojson_file, aoi_attributes, enhanced_geojson_path)
    
    print(f"\nFiles saved:")
    print(f"  - {output_dir}/class2_landcover_summary.csv")
    print(f"  - {output_dir}/all_classes_percentage_summary.csv")
    print(f"  - {output_dir}/detailed_analysis.csv")
    print(f"  - {enhanced_geojson_path}")
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"AOIs processed: {detailed_df['AOI_Name'].nunique()}")
    print(f"Years processed: {sorted(detailed_df['Year'].unique())}")
    print(f"Classes found: {sorted(detailed_df['Class'].unique())}")
    print(f"Total polygon area: {detailed_df.groupby('AOI_Name')['Polygon_Area_km2'].first().sum():.2f} km²")
    
    # Class 2 statistics
    class2_data = detailed_df[detailed_df['Class'] == 2]
    if not class2_data.empty:
        print(f"\nClass 2 Statistics:")
        print(f"  Average percentage across all AOIs/years: {class2_data['Percentage'].mean():.2f}%")
        print(f"  Max percentage: {class2_data['Percentage'].max():.2f}%")
        print(f"  Min percentage: {class2_data['Percentage'].min():.2f}%")
    
    return all_classes_df, class2_summary_df, detailed_df, enhanced_gdf

if __name__ == "__main__":
    main()
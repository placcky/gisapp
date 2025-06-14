# Third Code to RUN 
# This Python codes runs the analysis, still needs some refinements !! UNDER DEVELOPMENT!!

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from rasterio.mask import mask
import pandas as pd
from collections import defaultdict
import warnings
import os
import zipfile
from pathlib import Path
warnings.filterwarnings('ignore')

class MultiRasterTreeCoverAnalyzer:
    def __init__(self, raster_folder, geojson_path):
        """
        Initialize the analyzer with multiple rasters and GeoJSON regions
        
        Parameters:
        raster_folder (str):"./salzburg_raster"
        geojson_path (str): "./salzburg_AUT2.geojson"
        """
        self.raster_folder = raster_folder
        self.geojson_path = geojson_path
        self.raster_files = []
        self.regions_gdf = None
        self.all_results = []
        self.summary_results = pd.DataFrame()

    def find_tiff_files(self):
        """Find all TIFF files inside ZIP archives in the specified folder"""
        folder_path = Path(self.raster_folder)
        self.raster_files = []

        zip_files = list(folder_path.glob("*.zip"))

        print(f"Found {len(zip_files)} ZIP files:")

        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file_info in zip_ref.infolist():
                        if file_info.filename.lower().endswith((".tif", ".tiff")):
                            # Extract to a temporary path
                            extract_path = folder_path / f"__temp_extracted__" / zip_path.stem
                            extract_path.mkdir(parents=True, exist_ok=True)
                            zip_ref.extract(file_info, extract_path)
                            extracted_file = extract_path / file_info.filename
                            self.raster_files.append(extracted_file)
            except Exception as e:
                print(f"Error reading {zip_path.name}: {e}")

        print(f"Extracted and found {len(self.raster_files)} TIFF files:")
        for f in self.raster_files:
            print(f"  - {f}")
        
        if not self.raster_files:
            print("No TIFF files found inside ZIP archives!")
            
    def load_regions(self):
        """Load regions from GeoJSON file"""
        try:
            self.regions_gdf = gpd.read_file(self.geojson_path)
            print(f"Loaded {len(self.regions_gdf)} regions from GeoJSON")
            print(f"Regions CRS: {self.regions_gdf.crs}")
            
            # Display region names/IDs if available
            if 'name' in self.regions_gdf.columns:
                print("Region names:", list(self.regions_gdf['name']))
            elif 'id' in self.regions_gdf.columns:
                print("Region IDs:", list(self.regions_gdf['id']))
            else:
                print("Regions will be numbered 0 to", len(self.regions_gdf)-1)
                
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            
    def calculate_zonal_statistics_for_raster(self, raster_path, target_value=2):
        """
        Calculate zonal statistics for a single raster
        
        Parameters:
        raster_path (str): Path to the raster file
        target_value (int): Pixel value to analyze (default: 2 for tree cover loss)
        """
        results = []
        raster_name = Path(raster_path).stem
        
        print(f"\nProcessing raster: {raster_name}")
        
        try:
            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                
                # Reproject regions to match raster CRS if needed
                regions_projected = self.regions_gdf
                if self.regions_gdf.crs != raster_crs:
                    print(f"Reprojecting regions from {self.regions_gdf.crs} to {raster_crs}")
                    regions_projected = self.regions_gdf.to_crs(raster_crs)
                
                for idx, region in regions_projected.iterrows():
                    try:
                        # Get region identifier
                        if 'name' in region:
                            region_name = region['name']
                        elif 'id' in region:
                            region_name = str(region['id'])
                        else:
                            region_name = f'Region_{idx}'
                        
                        # Mask raster with region geometry
                        geom = [region.geometry.__geo_interface__]
                        masked_data, masked_transform = mask(src, geom, crop=True, nodata=src.nodata)
                        masked_data = masked_data[0]  # Get first band
                        
                        # Skip if no data in this region
                        if masked_data.size == 0:
                            continue
                        
                        # Calculate statistics for target value (tree cover loss)
                        loss_pixels = np.sum(masked_data == target_value)
                        
                        # Handle nodata values properly
                        if src.nodata is not None:
                            valid_pixels = np.sum(masked_data != src.nodata)
                        else:
                            valid_pixels = masked_data.size
                        
                        if valid_pixels == 0:
                            continue
                        
                        # Calculate area (assuming pixels represent area units)
                        pixel_area = abs(masked_transform.a * masked_transform.e)
                        loss_area = loss_pixels * pixel_area
                        total_area = valid_pixels * pixel_area
                        
                        # Calculate percentage
                        loss_percentage = (loss_pixels / valid_pixels * 100) if valid_pixels > 0 else 0
                        
                        # Store results
                        region_result = {
                            'raster_name': raster_name,
                            'region_id': idx,
                            'region_name': region_name,
                            'total_pixels': valid_pixels,
                            'loss_pixels': loss_pixels,
                            'total_area': total_area,
                            'loss_area': loss_area,
                            'loss_percentage': loss_percentage,
                            'pixel_area': pixel_area
                        }
                        results.append(region_result)
                        
                    except Exception as e:
                        print(f"Error processing region {idx} in {raster_name}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error opening raster {raster_path}: {e}")
            
        return results
    
    def process_all_rasters(self, target_value=2):
        """Process all rasters and calculate statistics"""
        if not self.raster_files:
            self.find_tiff_files()
        if self.regions_gdf is None:
            self.load_regions()
            
        self.all_results = []
        
        for raster_file in self.raster_files:
            raster_results = self.calculate_zonal_statistics_for_raster(raster_file, target_value)
            self.all_results.extend(raster_results)
        
        # Convert to DataFrame
        if self.all_results:
            self.summary_results = pd.DataFrame(self.all_results)
            print(f"\nTotal analysis completed: {len(self.all_results)} region-raster combinations")
        else:
            print("No results generated!")
            
        return self.summary_results
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        if self.summary_results.empty:
            print("No results to summarize")
            return
            
        print("\n" + "="*50)
        print("MULTI-RASTER TREE COVER LOSS SUMMARY")
        print("="*50)
        
        # Overall statistics
        total_regions = self.summary_results['region_name'].nunique()
        total_rasters = self.summary_results['raster_name'].nunique()
        total_combinations = len(self.summary_results)
        
        print(f"Analysis Overview:")
        print(f"  • Total regions analyzed: {total_regions}")
        print(f"  • Total rasters processed: {total_rasters}")
        print(f"  • Total region-raster combinations: {total_combinations}")
        
        # Area statistics
        total_area_analyzed = self.summary_results['total_area'].sum()
        total_loss_area = self.summary_results['loss_area'].sum()
        overall_loss_percentage = (total_loss_area / total_area_analyzed * 100) if total_area_analyzed > 0 else 0
        
        print(f"\nArea Statistics:")
        print(f"  • Total area analyzed: {total_area_analyzed:.2f} sq units")
        print(f"  • Total loss area: {total_loss_area:.2f} sq units")
        print(f"  • Overall loss percentage: {overall_loss_percentage:.2f}%")
        
        # Loss percentage statistics
        print(f"\nLoss Percentage Statistics:")
        print(f"  • Average loss percentage: {self.summary_results['loss_percentage'].mean():.2f}%")
        print(f"  • Maximum loss percentage: {self.summary_results['loss_percentage'].max():.2f}%")
        print(f"  • Minimum loss percentage: {self.summary_results['loss_percentage'].min():.2f}%")
        print(f"  • Standard deviation: {self.summary_results['loss_percentage'].std():.2f}%")
        
        # By raster summary
        print(f"\nSummary by Raster:")
        raster_summary = self.summary_results.groupby('raster_name').agg({
            'loss_area': 'sum',
            'total_area': 'sum',
            'loss_percentage': 'mean'
        }).round(2)
        raster_summary['overall_loss_pct'] = (raster_summary['loss_area'] / raster_summary['total_area'] * 100).round(2)
        print(raster_summary)
        
        # By region summary
        print(f"\nSummary by Region:")
        region_summary = self.summary_results.groupby('region_name').agg({
            'loss_area': 'sum',
            'total_area': 'sum',
            'loss_percentage': 'mean'
        }).round(2)
        region_summary['overall_loss_pct'] = (region_summary['loss_area'] / region_summary['total_area'] * 100).round(2)
        print(region_summary)
        
        return self.summary_results.describe()
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for multi-raster analysis"""
        if self.summary_results.empty:
            print("No results to visualize")
            return
            
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Loss percentage by raster and region (heatmap)
        ax1 = plt.subplot(3, 3, 1)
        pivot_data = self.summary_results.pivot_table(
            values='loss_percentage', 
            index='region_name', 
            columns='raster_name', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='Reds', ax=ax1, cbar_kws={'label': 'Loss %'})
        ax1.set_title('Tree Cover Loss % by Region and Raster')
        ax1.set_xlabel('Raster')
        ax1.set_ylabel('Region')
        
        # 2. Total loss area by raster
        ax2 = plt.subplot(3, 3, 2)
        raster_loss = self.summary_results.groupby('raster_name')['loss_area'].sum()
        bars = ax2.bar(range(len(raster_loss)), raster_loss.values, color='darkred', alpha=0.7)
        ax2.set_xlabel('Raster')
        ax2.set_ylabel('Total Loss Area')
        ax2.set_title('Total Loss Area by Raster')
        ax2.set_xticks(range(len(raster_loss)))
        ax2.set_xticklabels(raster_loss.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Total loss area by region
        ax3 = plt.subplot(3, 3, 3)
        region_loss = self.summary_results.groupby('region_name')['loss_area'].sum()
        bars = ax3.bar(range(len(region_loss)), region_loss.values, color='forestgreen', alpha=0.7)
        ax3.set_xlabel('Region')
        ax3.set_ylabel('Total Loss Area')
        ax3.set_title('Total Loss Area by Region')
        ax3.set_xticks(range(len(region_loss)))
        ax3.set_xticklabels(region_loss.index, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution of loss percentages
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(self.summary_results['loss_percentage'], bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Tree Cover Loss Percentage')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Loss Percentages')
        ax4.grid(True, alpha=0.3)
        
        # 5. Box plot by raster
        ax5 = plt.subplot(3, 3, 5)
        raster_names = self.summary_results['raster_name'].unique()
        box_data = [self.summary_results[self.summary_results['raster_name'] == raster]['loss_percentage'].values 
                   for raster in raster_names]
        bp = ax5.boxplot(box_data, labels=raster_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax5.set_ylabel('Loss Percentage')
        ax5.set_title('Loss Percentage Distribution by Raster')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Box plot by region
        ax6 = plt.subplot(3, 3, 6)
        region_names = self.summary_results['region_name'].unique()
        box_data = [self.summary_results[self.summary_results['region_name'] == region]['loss_percentage'].values 
                   for region in region_names]
        bp = ax6.boxplot(box_data, labels=region_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax6.set_ylabel('Loss Percentage')
        ax6.set_title('Loss Percentage Distribution by Region')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # 7. Scatter plot: Total area vs Loss area
        ax7 = plt.subplot(3, 3, 7)
        scatter = ax7.scatter(self.summary_results['total_area'], self.summary_results['loss_area'], 
                            c=self.summary_results['loss_percentage'], cmap='Reds', 
                            s=60, alpha=0.7, edgecolors='black')
        ax7.set_xlabel('Total Area')
        ax7.set_ylabel('Loss Area')
        ax7.set_title('Loss Area vs Total Area')
        plt.colorbar(scatter, ax=ax7, label='Loss %')
        ax7.grid(True, alpha=0.3)
        
        # 8. Summary statistics table
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('tight')
        ax8.axis('off')
        
        summary_data = [
            ['Total Regions', len(self.summary_results['region_name'].unique())],
            ['Total Rasters', len(self.summary_results['raster_name'].unique())],
            ['Total Combinations', len(self.summary_results)],
            ['Total Area', f"{self.summary_results['total_area'].sum():.0f}"],
            ['Total Loss Area', f"{self.summary_results['loss_area'].sum():.0f}"],
            ['Avg Loss %', f"{self.summary_results['loss_percentage'].mean():.2f}%"],
            ['Max Loss %', f"{self.summary_results['loss_percentage'].max():.2f}%"],
            ['Min Loss %', f"{self.summary_results['loss_percentage'].min():.2f}%"]
        ]
        
        table = ax8.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax8.set_title('Summary Statistics')
        
        # 9. Top regions/rasters by loss
        ax9 = plt.subplot(3, 3, 9)
        top_combinations = self.summary_results.nlargest(10, 'loss_area')
        labels = [f"{row['region_name']}\n({row['raster_name']})" for _, row in top_combinations.iterrows()]
        y_pos = range(len(labels))
        ax9.barh(y_pos, top_combinations['loss_area'], color='crimson', alpha=0.7)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(labels, fontsize=8)
        ax9.set_xlabel('Loss Area')
        ax9.set_title('Top 10 Region-Raster Combinations by Loss Area')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Multi-Raster Tree Cover Loss Analysis Results', fontsize=18, y=0.98)
        plt.show()
        
        return fig
    
    def cleanup_temp_extracted(self):
        temp_path = Path(self.raster_folder) / "__temp_extracted__"
        if temp_path.exists():
            import shutil
            shutil.rmtree(temp_path)
            print("Temporary extracted files removed.")


# Main analysis function
def analyze_multiple_rasters_tree_cover_loss(raster_folder, geojson_path, target_value=2):
    """
    Main function to analyze tree cover loss across multiple rasters and regions
    
    Parameters:
    raster_folder (str): Path to folder containing TIFF files
    geojson_path (str): Path to GeoJSON file with regions
    target_value (int): Pixel value representing tree cover loss (default: 2)
    """
    
    # Initialize analyzer
    analyzer = MultiRasterTreeCoverAnalyzer(raster_folder, geojson_path)
    
    # Find TIFF files
    analyzer.find_tiff_files()
    
    # Load regions
    analyzer.load_regions()
    
    # Process all rasters
    results = analyzer.process_all_rasters(target_value)
    
    # Generate summary
    summary = analyzer.generate_summary_statistics()
    
    # Create visualizations
    fig = analyzer.create_comprehensive_visualizations()
    
    return analyzer, results, summary

# Example usage:

# Define your paths
raster_folder = "./salzburg_raster"  # Folder containing all your .tif files
geojson_file = "./salzburg_AUT2.geojson"  # Your GeoJSON file with regions

# Run the analysis
analyzer, results, summary = analyze_multiple_rasters_tree_cover_loss(raster_folder, geojson_file)

# Access results
print("\nDetailed Results:")
print(results[['raster_name', 'region_name', 'loss_percentage', 'loss_area']].to_string(index=False))

# Save results to CSV
results.to_csv('tree_cover_loss_results.csv', index=False)
print("\nResults saved to 'tree_cover_loss_results.csv'")

        let map;
        let geojsonLayer;

        function initializeMap() {
            map = L.map('map').setView([47.5, 13.2], 8);

            // Base layers
            const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors, Tiles by HOT OSM'
            });

            const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: '&copy; Esri, Maxar, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community'
            });

            const cartoLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
            });

            // Add default layer
            osmLayer.addTo(map);

            // Controls
            const baseLayers = {
                "OpenStreetMap": osmLayer,
                "Satellite": satelliteLayer,
                "Light Theme": cartoLayer
            };

            L.control.layers(baseLayers, null, { position: 'bottomleft' }).addTo(map);
            L.control.scale({ position: 'bottomright', metric: true, imperial: false }).addTo(map);
        }

        function getColor(loss) {
            return loss > 4 ? '#8e44ad' :
                   loss > 3 ? '#e74c3c' :
                   loss > 2 ? '#e67e22' :
                   loss > 1 ? '#f1c40f' :
                              '#2ecc71';
        }

        function calculateTotalLoss(properties) {
            const start = properties['2018_Class2_Area_km2'];
            const end = properties['2024_Class2_Area_km2'];
            if (start && end && start > 0) {
                return ((start - end) / start * 100).toFixed(1);
            }
            return '0.0';
        }

        function style(feature) {
            const loss = parseFloat(calculateTotalLoss(feature.properties));
            return {
                fillColor: getColor(loss),
                weight: 2,
                opacity: 1,
                color: 'white',
                dashArray: '3',
                fillOpacity: 0.7
            };
        }

        function createPopupContent(feature) {
            const props = feature.properties;
            const regionName = props.NAME_2 || props.NAME_1 || 'Unknown';
            const totalArea = props.Polygon_Area_km2?.toFixed(1) || 'N/A';
            const forest2018 = props['2018_Class2_Area_km2']?.toFixed(1) || 'N/A';
            const forest2024 = props['2024_Class2_Area_km2']?.toFixed(1) || 'N/A';
            const rawChange = (props['2018_Class2_Area_km2'] - props['2024_Class2_Area_km2']).toFixed(2);
            const totalLoss = calculateTotalLoss(props);

            // Get years and forest % per year
            const years = [];
            const coverPercents = [];

            for (let year = 2018; year <= 2024; year++) {
                const area = props[`${year}_Class2_Area_km2`];
                if (area && props.Polygon_Area_km2) {
                    years.push(year);
                    const percent = (area / props.Polygon_Area_km2) * 100;
                    coverPercents.push(percent.toFixed(1));
                }
            }

            // Unique chart ID per feature
            const chartId = `chart-${Math.random().toString(36).substring(2, 10)}`;

            setTimeout(() => {
                const ctx = document.getElementById(chartId);
                if (ctx) {
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: years,
                            datasets: [{
                                label: 'Forest Cover (%)',
                                data: coverPercents,
                                borderColor: '#27ae60',
                                backgroundColor: 'rgba(39, 174, 96, 0.2)',
                                tension: 0.3,
                                fill: true,
                                pointRadius: 4,
                                pointBackgroundColor: '#27ae60'
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => `${ctx.raw}%`
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100,
                                    ticks: { stepSize: 20 }
                                }
                            }
                        }
                    });
                }
            }, 0);

            return `
                <div class="popup-content">
                    <div class="popup-title">${regionName} – Total Area ${totalArea} km²</div>
                    <div class="stats-grid">
                        <div class="stat-item"><div class="stat-value">${totalLoss}</div><div class="stat-label">Tree cover Loss (%)</div></div>
                        <div class="stat-item"><div class="stat-value">${rawChange}</div><div class="stat-label">Raw Change in tree cover (km²)</div></div>
                        <div class="stat-item"><div class="stat-value">${forest2018}</div><div class="stat-label">Tree cover 2018 (km²)</div></div>
                        <div class="stat-item"><div class="stat-value">${forest2024}</div><div class="stat-label">Tree cover 2024 (km²)</div></div>
                    </div>
                    <div class="chart-container">
                        <canvas id="${chartId}"></canvas>
                    </div>
                </div>`;
        }

        function onEachFeature(feature, layer) {
            layer.on({
                mouseover: e => {
                    e.target.setStyle({ weight: 4, color: '#2c3e50', fillOpacity: 0.9 });
                    e.target.bringToFront();
                },
                mouseout: e => geojsonLayer.resetStyle(e.target),
                click: e => {
                    const popupContent = createPopupContent(feature);
                    layer.bindPopup(popupContent, { maxWidth: 450 }).openPopup();
                }
            });
        }

        async function loadGeoJSONData() {
    try {
        console.log('Loading GeoJSON data...');
        // Try multiple possible paths for your GeoJSON file
        const possiblePaths = [
            './results/salzburg_enhanced.geojson',
            './salzburg_enhanced.geojson',
            'salzburg_enhanced.geojson',
            './data/salzburg_enhanced.geojson'
        ];
        
        let data = null;
        let successfulPath = null;
        
        for (const path of possiblePaths) {
            try {
                console.log(`Trying path: ${path}`);
                const res = await fetch(path);
                if (res.ok) {
                    data = await res.json();
                    successfulPath = path;
                    console.log(`Successfully loaded from: ${path}`);
                    break;
                }
            } catch (err) {
                console.log(`Failed to load from ${path}:`, err.message);
            }
        }
        
        if (!data) {
            throw new Error('GeoJSON file not found in any expected location');
        }

        geojsonLayer = L.geoJSON(data, {
            style,
            onEachFeature
        }).addTo(map);

        map.fitBounds(geojsonLayer.getBounds());
        
        // Create summary chart for info modal
        createSummaryChart(data);
        
        console.log('GeoJSON data loaded successfully');

    } catch (err) {
        console.error('GeoJSON Load Error:', err);
        alert(`Failed to load GeoJSON: ${err.message}`);
    }
}

        function showInfoModal() {
            document.getElementById('infoModal').style.display = 'flex';
        }

        function hideInfoModal() {
            document.getElementById('infoModal').style.display = 'none';
        }

        function createSummaryChart(geojsonData) {
    const ctx = document.getElementById('summaryChart');
    if (!ctx || !geojsonData) return;

    const years = [2018, 2019, 2020, 2021, 2022, 2023, 2024];
    const regions = [];
    const datasets = [];
    
    // Colors for each region
    const colors = ['#e74c3c', '#f39c12', '#f1c40f', '#27ae60', '#3498db', '#9b59b6'];
    
    geojsonData.features.forEach((feature, index) => {
        const props = feature.properties;
        const regionName = props.NAME_2 || props.NAME_1 || `Region ${index + 1}`;
        regions.push(regionName);
        
        const coverageData = [];
        years.forEach(year => {
            const area = props[`${year}_Class2_Area_km2`];
            const totalArea = props.Polygon_Area_km2;
            if (area && totalArea) {
                const percentage = (area / totalArea) * 100;
                coverageData.push(percentage.toFixed(1));
            } else {
                coverageData.push(0);
            }
        });
        
        datasets.push({
            label: regionName,
            data: coverageData,
            borderColor: colors[index % colors.length],
            backgroundColor: colors[index % colors.length] + '20',
            tension: 0.3,
            fill: false,
            pointRadius: 3
        });
    });

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { fontSize: 12 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        stepSize: 10,
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Tree Cover (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                }
            }
        }
    });
}

        function handleResize() {
            if (map) {
                setTimeout(() => {
                    map.invalidateSize();
                }, 100);
            }
        }

        // Event listeners
        window.addEventListener('resize', handleResize);
        window.addEventListener('orientationchange', handleResize);

        document.addEventListener('DOMContentLoaded', () => {
            initializeMap();
            loadGeoJSONData();
            showInfoModal();
            setTimeout(() => {
                document.querySelector('.container').classList.add('loaded');
            }, 300);
        });
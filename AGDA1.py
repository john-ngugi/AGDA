import streamlit as st
import os
import re
import numpy as np
import json
import tempfile
from werkzeug.utils import secure_filename
import geopandas as gpd
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
import requests
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium 
import time

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'shp', 'dbf', 'shx', 'prj', 'geojson', 'kml', 'json'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create upload folder if doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def add_typing_effect(text, speed=0.02):
    """
    Add a typing effect to text in Streamlit
    
    Args:
        text (str): The text to display with a typing effect
        speed (float): Delay between characters in seconds
    """
    placeholder = st.empty()
    
    # For markdown formatting, handle by sections
    sections = text.split('\n\n')
    current_text = ""
    
    for section in sections:
        # For code blocks, add them all at once to preserve formatting
        if section.startswith('```') and section.endswith('```'):
            current_text += section + '\n\n'
            placeholder.markdown(current_text)
            time.sleep(speed * 10)  # Slightly longer pause after code blocks
            continue
            
        # For normal text, add word by word
        words = section.split(' ')
        for i, word in enumerate(words):
            current_text += word + ' '
            # Update every few words for smoother effect
            if i % 3 == 0 or i == len(words) - 1:
                placeholder.markdown(current_text)
                time.sleep(speed)
        
        # Add the section separator
        current_text += '\n\n'
        placeholder.markdown(current_text)
        time.sleep(speed * 2)  # Slightly longer pause between sections




# DeepSeek LLM Integration
class DeepSeekLLM(BaseLLM):
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.7
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        responses = []
        for prompt in prompts:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    **kwargs
                },
                headers=headers,
            ).json()
            responses.append(response["choices"][0]["message"]["content"])
        
        return LLMResult(generations=[[{"text": r}] for r in responses])
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"

# Initialize LLM (with API key in Streamlit secrets)
@st.cache_resource
def get_llm():
    # In production, use st.secrets instead of hardcoded key
    api_key = st.secrets["DEEPSEEK_API_KEY"]
    # something # Replace with st.secrets in production
    return DeepSeekLLM(api_key=api_key)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_gis_file(file_path: str, file_type: str):
    """Load GIS file into GeoDataFrame"""
    if file_type in ['geojson', 'json']:
        return gpd.read_file(file_path, driver='GeoJSON')
    elif file_type == 'kml':
        return gpd.read_file(file_path, driver='KML')
    else:  # Shapefile
        return gpd.read_file(file_path)

# Analysis Functions
def generate_data_summary(gdf):
    """Generate comprehensive summary of the GeoDataFrame"""
    summary = {
        'overview': {
            'crs': str(gdf.crs),
            'geometry_type': list(gdf.geom_type.unique()),
            'total_features': len(gdf),
            'columns': list(gdf.columns),
            'bounds': gdf.total_bounds.tolist()
        },
        'statistics': {},
        'spatial_properties': {},
        'data_quality': {
            'missing_values': gdf.isna().sum().to_dict(),
            'duplicate_features': len(gdf) - len(gdf.drop_duplicates())
        }
    }
    
    # Spatial properties
    if not gdf.empty and hasattr(gdf, 'geometry'):
        summary['spatial_properties'] = {
            'area_stats': calculate_area_stats(gdf),
            'length_stats': calculate_length_stats(gdf),
            'centroid': calculate_centroid(gdf)
        }
    
    # Column statistics
    for col in gdf.columns:
        if col != 'geometry':
            col_data = gdf[col]
            if np.issubdtype(col_data.dtype, np.number):
                summary['statistics'][col] = calculate_numeric_stats(col_data)
            else:
                summary['statistics'][col] = calculate_categorical_stats(col_data)
    
    # Convert numpy types before returning
    return convert_numpy_types(summary)

def calculate_centroid(gdf):
    """Safely calculate centroid for single and multi-part geometries"""
    try:
        # Convert to single part geometries if needed
        if any(t in ['MultiPolygon', 'MultiLineString', 'MultiPoint'] for t in gdf.geom_type.unique()):
            # Explode multi-part geometries
            exploded = gdf.explode(index_parts=True)
            # Get representative point for each geometry
            points = exploded.geometry.representative_point()
        else:
            points = gdf.geometry.centroid
        
        # Calculate mean centroid
        x_coords = [p.x for p in points if not p.is_empty]
        y_coords = [p.y for p in points if not p.is_empty]
        
        if x_coords and y_coords:
            return [np.mean(x_coords), np.mean(y_coords)]
        return None
    except Exception as e:
        st.error(f"Centroid calculation error: {str(e)}")
        return None

def calculate_numeric_stats(series):
    """Calculate detailed statistics for numeric columns"""
    clean_series = series.dropna()
    stats = {
        'type': 'numeric',
        'count': clean_series.count(),
        'mean': clean_series.mean(),
        'std': clean_series.std(),
        'min': clean_series.min(),
        'percentiles': {
            '25%': clean_series.quantile(0.25),
            '50%': clean_series.quantile(0.5),
            '75%': clean_series.quantile(0.75),
            '90%': clean_series.quantile(0.9)
        },
        'max': clean_series.max(),
        'skewness': clean_series.skew(),
        'kurtosis': clean_series.kurtosis(),
        'missing_values': series.isna().sum(),
        'zeros': (series == 0).sum()
    }
    return convert_numpy_types(stats)

def calculate_categorical_stats(series):
    """Calculate statistics for categorical columns"""
    value_counts = series.value_counts()
    return {
        'type': 'categorical',
        'count': series.count(),
        'unique_values': len(value_counts),
        'top_values': value_counts.head(10).to_dict(),
        'missing_values': series.isna().sum()
    }

def calculate_area_stats(gdf):
    """Calculate area statistics for polygons"""
    if any(t in ['Polygon', 'MultiPolygon'] for t in gdf.geom_type.unique()):
        areas = gdf.geometry.area
        return {
            'min_area': areas.min(),
            'max_area': areas.max(),
            'mean_area': areas.mean(),
            'median_area': areas.median(),
            'total_area': areas.sum(),
            'unit': 'square meters'
        }
    return {}

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    return obj

def calculate_length_stats(gdf):
    """Calculate length statistics for lines"""
    if any(t in ['LineString', 'MultiLineString'] for t in gdf.geom_type.unique()):
        lengths = gdf.geometry.length
        return {
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'total_length': lengths.sum(),
            'unit': 'meters'
        }
    return {}

def detect_spatial_clusters(gdf, eps=0.1, min_samples=5):
    """Detect spatial clusters using DBSCAN"""
    try:
        if len(gdf) < min_samples:
            return None
            
        # Extract coordinates and scale them
        coords = np.array([(geom.x, geom.y) for geom in gdf.geometry.centroid])
        scaled_coords = StandardScaler().fit_transform(coords)
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_coords)
        labels = db.labels_
        
        # Count clusters (ignore noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return {
            'n_clusters': n_clusters,
            'noise_points': list(labels).count(-1),
            'cluster_sizes': [list(labels).count(i) for i in range(n_clusters)],
            'cluster_examples': get_cluster_examples(gdf, labels)
        }
    except Exception as e:
        st.error(f"Cluster detection failed: {str(e)}")
        return None

def get_cluster_examples(gdf, labels, n_examples=3):
    """Get example features from each cluster"""
    examples = {}
    unique_labels = set(labels) - {-1}
    
    for label in unique_labels:
        cluster_samples = gdf[labels == label].sample(min(n_examples, sum(labels == label)))
        examples[label] = [
            {col: sample[col] for col in cluster_samples.columns if col != 'geometry'}
            for _, sample in cluster_samples.iterrows()
        ]
    
    return examples

# Visualization Functions
import branca.colormap as cm

import folium
import numpy as np
import pandas as pd
import branca.colormap as cm

def create_attribute_map(gdf, attribute, bins=4):
    """Create Folium choropleth map for a specific attribute using quantiles."""

    # Reproject to WGS84 if needed
    if gdf.crs and gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs("EPSG:4326")

    # Center map roughly to Kenya
    m = folium.Map(location=[-1, 36], zoom_start=9)

    if attribute in gdf.columns:
        if np.issubdtype(gdf[attribute].dtype, np.number):
            # Compute quantile bins
            gdf = gdf.copy()
            try:
                gdf["bin"] = pd.qcut(gdf[attribute], q=bins, duplicates="drop")
            except ValueError:
                gdf["bin"] = pd.cut(gdf[attribute], bins=bins)

            # Create a color map for bins
            unique_bins = gdf["bin"].unique().categories if hasattr(gdf["bin"].unique(), 'categories') else sorted(gdf["bin"].unique())
            colors = cm.linear.Viridis_09.scale(0, len(unique_bins)).to_step(len(unique_bins)).colors
            bin_color_map = {str(b): colors[i] for i, b in enumerate(unique_bins)}

            # Add GeoJson with quantile-based coloring
            folium.GeoJson(
                gdf,
                style_function=lambda feature: {
                    'fillColor': bin_color_map.get(str(feature['properties'].get("bin")), 'gray'),
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=folium.GeoJsonTooltip(fields=[attribute]),
                name=attribute
            ).add_to(m)

            # Create legend manually
            legend_html = f'<div style="position: fixed; bottom: 10px; left: 10px; z-index: 9999; background-color: white; padding: 10px; border:2px solid grey;"><b>{attribute} (Quantiles)</b><br>'
            for b in unique_bins:
                legend_html += f'<i style="background:{bin_color_map[str(b)]};width:10px;height:10px;display:inline-block;margin-right:5px;"></i>{b}<br>'
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))

        else:
            # Handle categorical data
            categories = gdf[attribute].dropna().unique()
            color_map = {cat: color for cat, color in zip(categories, cm.Set1.colors)}

            folium.GeoJson(
                gdf,
                style_function=lambda feature: {
                    'fillColor': color_map.get(feature['properties'][attribute], 'gray'),
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=folium.GeoJsonTooltip(fields=[attribute]),
                name=attribute
            ).add_to(m)

            legend_html = f'<div style="position: fixed; bottom: 10px; left: 10px; z-index: 9999; background-color: white; padding: 10px; border:2px solid grey;"><b>{attribute}</b><br>'
            for cat, color in color_map.items():
                legend_html += f'<i style="background:{color};width:10px;height:10px;display:inline-block;margin-right:5px;"></i>{cat}<br>'
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))

    else:
        folium.GeoJson(
            gdf,
            style_function=lambda feature: {
                'fillColor': 'lightblue',
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.6,
            },
            name="Default Layer"
        ).add_to(m)

    return m

def create_histogram(gdf, attribute):
    """Create enhanced histogram with stats"""
    if attribute not in gdf.columns or not np.issubdtype(gdf[attribute].dtype, np.number):
        return None
    
    values = gdf[attribute].dropna()
    
    # Calculate statistics
    mean = values.mean()
    median = values.median()
    std = values.std()
    skewness = values.skew()
    kurtosis = values.kurtosis()
    
    # Create histogram with Plotly
    fig = px.histogram(
        gdf, 
        x=attribute,
        title=f'Distribution of {attribute} (Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f})',
        marginal="box"
    )
    
    # Add mean and median lines
    fig.add_vline(x=mean, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean:.2f}", annotation_position="top right")
    fig.add_vline(x=median, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median:.2f}", annotation_position="top left")
    
    # Add standard deviation lines
    fig.add_vline(x=mean-std, line_dash="dot", line_color="blue")
    fig.add_vline(x=mean+std, line_dash="dot", line_color="blue")
    
    return fig

def create_scatter_plot(gdf, x_attr, y_attr):
    """Create scatter plot with regression line"""
    if x_attr not in gdf.columns or y_attr not in gdf.columns:
        return None
    
    # Filter non-null values for both attributes
    df_filtered = gdf[[x_attr, y_attr]].dropna()
    
    # Create scatter plot with regression line
    fig = px.scatter(
        df_filtered, 
        x=x_attr, 
        y=y_attr,
        trendline="ols",
        trendline_color_override="red",
        title=f'{y_attr} vs {x_attr}'
    )
    
    # Get trendline equation and r-squared
    import statsmodels.api as sm
    X = df_filtered[x_attr]
    y = df_filtered[y_attr]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # Add equation and r-squared to title
    intercept = model.params[0]
    slope = model.params[1]
    r_squared = model.rsquared
    fig.update_layout(
        title=f'{y_attr} vs {x_attr}<br>y = {intercept:.2f} + {slope:.2f}x (r² = {r_squared:.2f})'
    )
    
    return fig

def create_box_plot(gdf, attribute):
    """Create box plot for an attribute"""
    if attribute not in gdf.columns or not np.issubdtype(gdf[attribute].dtype, np.number):
        return None
    
    fig = px.box(
        gdf,
        y=attribute,
        title=f'Box Plot of {attribute}'
    )
    
    # Add mean marker
    mean_val = gdf[attribute].mean()
    fig.add_trace(
        go.Scatter(
            x=[0], 
            y=[mean_val],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Mean'
        )
    )
    
    return fig

# Query Handling Functions
def extract_attribute_from_query(query, gdf_columns):
    """Extract attribute name from query"""
    # First try to find exact column matches
    for col in gdf_columns:
        if col.lower() in query.lower():
            return col
    
    # Then try pattern matching
    patterns = [
        r"(?:histogram|map|chart|graph|plot|distribution|analyze|show|display) (?:of|for) (\w+)",
        r"(?:show|display) (\w+) (?:histogram|map|chart|graph|plot|distribution)",
        r"(\w+) (?:histogram|map|chart|graph|plot|distribution)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            potential_attr = match.group(1)
            if potential_attr in gdf_columns:
                return potential_attr
    
    return None

def extract_attributes_from_query(query, gdf_columns, n=2):
    """Extract multiple attributes from query"""
    # First try to find exact column matches
    found = [col for col in gdf_columns if col.lower() in query.lower()]
    if len(found) >= n:
        return found[:n]
    
    # Then try pattern matching
    patterns = [
        r"(?:relationship|correlation|compare|between) (\w+) (?:and|&) (\w+)",
        r"(\w+) (?:vs|versus|and|&) (\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            attrs = [match.group(1), match.group(2)]
            if all(a in gdf_columns for a in attrs):
                return attrs
    
    return None

def process_gis_file(file):
    """Process uploaded GIS file and return GeoDataFrame"""
    try:
        file_name = file.name
        file_ext = file_name.rsplit('.', 1)[1].lower()
        
        # Create a temporary file to store the uploaded data
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
            
        # Load based on file type
        if file_ext in ['geojson', 'json']:
            gdf = gpd.read_file(temp_file_path, driver='GeoJSON')
            file_type = 'geojson'
        elif file_ext == 'kml':
            gdf = gpd.read_file(temp_file_path, driver='KML')
            file_type = 'kml'
        elif file_ext == 'shp':
            gdf = gpd.read_file(temp_file_path)
            file_type = 'shp'
        else:
            return None, None, f"Unsupported file type: {file_ext}"
            
        # Get metadata
        metadata = {
            'crs': str(gdf.crs),
            'geometry_type': gdf.geom_type.unique().tolist(),
            'features_count': len(gdf),
            'columns': list(gdf.columns),
            'bounds': gdf.total_bounds.tolist()
        }
        
        return gdf, file_type, metadata
        
    except Exception as e:
        return None, None, f"Error processing file: {str(e)}"
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def analyze_gis_data(gdf, query: str):
    """Perform enhanced analysis on GeoDataFrame based on query"""
    try:
        llm = get_llm()
        
        # Generate comprehensive data summary
        summary = generate_data_summary(gdf)
        
        # Check for specific analysis requests
        if "cluster" in query.lower() or "group" in query.lower():
            cluster_info = detect_spatial_clusters(gdf)
            if cluster_info:
                summary['spatial_clusters'] = cluster_info
        
        if "correlation" in query.lower() or "relationship" in query.lower():
            corr_matrix = gdf.select_dtypes(include=np.number).corr()
            summary['correlations'] = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False).to_dict()
        
        # Prepare AI prompt with enhanced context
        prompt = f"""
        You are a professional GIS data scientist analyzing geospatial data. 
        You perform various spatial analysis and visualization tasks.
        Here is a comprehensive summary of the dataset:

        {json.dumps(summary, indent=2)}

        User Question: {query}

        Provide a detailed response including:
        1. Key statistics and patterns relevant to the question
        2. Spatial analysis if applicable
        3. Data quality considerations
        4. Suggested visualizations
        5. Recommended next steps for analysis
        6. Any limitations or caveats

        Structure your response with clear sections and use markdown formatting.
        """
        
        # Get AI response
        response = llm.generate([prompt])
        return response.generations[0][0].text
        
    except Exception as e:
        return f"Analysis error: {str(e)}"

def query_gis_data(query: str, gdf):
    """Handle enhanced GIS data queries and return the appropriate results"""
    try:
        # For basic mapping and visualization
        if "map" in query.lower():
            attr = extract_attribute_from_query(query, gdf.columns)
            if attr:
                return "map", attr
        
        if "histogram" in query.lower() or "distribution" in query.lower():
            attr = extract_attribute_from_query(query, gdf.columns)
            if attr:
                return "histogram", attr
                
        if "box plot" in query.lower():
            attr = extract_attribute_from_query(query, gdf.columns)
            if attr:
                return "boxplot", attr
            
        if "scatter" in query.lower() or "relationship between" in query.lower() or "correlation" in query.lower():
            attrs = extract_attributes_from_query(query, gdf.columns, 2)
            if attrs and len(attrs) == 2:
                return "scatter", attrs
        
        # Default to comprehensive analysis
        analysis_result = analyze_gis_data(gdf, query)
        return "text", analysis_result
    
    except Exception as e:
        return "error", f"Error processing query: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(page_title="AGDA", page_icon="🌍", layout="wide")
    
    st.title("Hello 👋, I'm AGDA")
    st.markdown("""
    Upload your geospatial data files (GeoJSON, Shapefile, KML) and ask questions about the data.
    The application will analyze the data and provide visualizations and insights.
    """)
    
    # File upload section
    st.header("📤 Upload GIS Data")
    uploaded_file = st.file_uploader(
        "Choose a GIS file", 
        type=list(ALLOWED_EXTENSIONS), 
        help="Upload GeoJSON, Shapefile, or KML"
    )
    
    # Initialize session state for storing the GeoDataFrame
    if 'gdf' not in st.session_state:
        st.session_state.gdf = None
    if 'file_type' not in st.session_state:
        st.session_state.file_type = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.spinner("Processing GIS file..."):
            gdf, file_type, result = process_gis_file(uploaded_file)
            
            if isinstance(result, dict):  # Success
                st.session_state.gdf = gdf
                st.session_state.file_type = file_type
                st.session_state.metadata = result
                
                # Display metadata
                st.success(f"File processed successfully: {uploaded_file.name}")
                st.subheader("📊 Dataset Overview")
                with st.expander("See attribute Labels"):    
                    # Show column names
                    st.write("**Available attributes:**")
                    col_list = [col for col in result['columns'] if col != 'geometry']
                    st.write(", ".join(col_list))
                    
                col1, col2 , col3= st.columns([1, 1, 1.5])
                with col1:
                    st.metric("Features", result['features_count'])
                    st.write(f"**Geometry type:** {', '.join(result['geometry_type'])}")
                with col2:
                    st.metric("Attributes", len(result['columns']) - 1 if 'geometry' in result['columns'] else len(result['columns']))
                    st.write(f"**CRS:** {result['crs']}")
                with col3:
                    with st.expander("See attributes Table"):
                       st.write(gdf)
                       

                
                # Simple map preview
                st.subheader("🗺️ Map Preview")
                try:
                    # Center map on data
                    m = folium.Map(location=[-1, 36], zoom_start=7,tiles="cartodb positron")
                    # Add GeoDataFrame to map
                    folium.GeoJson(gdf, name="GeoData").add_to(m)

                    with st.container():
                        show_map = st.toggle("Display map")

                        if show_map:
                            #Display the map
                            
                            st_data = st_folium(m, width="100%", height=600)

                except Exception as e:
                    st.error(f"Could not create map preview: {str(e)}")
            else:  # Error
                st.error(result)
    
    # Query section
    st.header("❓ Ask Questions About Your Data")
    if st.session_state.gdf is not None:
        query = st.text_input(
            "Enter your question about the data",
            placeholder="e.g., 'Show me a map of population' or 'What are the key patterns in this dataset?'"
        )
        
        if query:
            with st.spinner("Analyzing..."):
                result_type, result_data = query_gis_data(query, st.session_state.gdf)
                
                # Display results based on type
                if result_type == "map":
                    st.subheader(f"Map of {result_data}")
                    fig = create_attribute_map(st.session_state.gdf, result_data)
                    st_data = st_folium(fig, width="100%", height=600)
                    
                elif result_type == "histogram":
                    st.subheader(f"Histogram of {result_data}")
                    fig = create_histogram(st.session_state.gdf, result_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not create histogram for attribute: {result_data}")
                    
                elif result_type == "boxplot":
                    st.subheader(f"Box Plot of {result_data}")
                    fig = create_box_plot(st.session_state.gdf, result_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not create box plot for attribute: {result_data}")
                    
                elif result_type == "scatter":
                    st.subheader(f"Scatter Plot: {result_data[1]} vs {result_data[0]}")
                    fig = create_scatter_plot(st.session_state.gdf, result_data[0], result_data[1])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not create scatter plot for the selected attributes")
                    
                elif result_type == "text":
                    # add_typing_effect(result_data)
                    st.markdown(result_data)
                    
                else:  # error
                    st.error(result_data)
    else:
        st.info("Please upload a GIS file first to enable data analysis.")
    
    # Footer
    st.markdown("---")
    st.caption("Made with ❤️ by mtaa wetu • Powered by streamlit, DeepSeek LLM")

if __name__ == "__main__":
    main()
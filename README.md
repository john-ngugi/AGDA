# 🌍 AGDA - Advanced GIS Data Analyzer

**Version 1.0 | Built by [Mtaa Wetu](https://mtaawetu.com) | Powered by Streamlit & DeepSeek**

Welcome to **AGDA (Advanced GIS Data Analyzer)** — an intelligent, user-friendly spatial data analysis agent built to empower communities, planners, and researchers with powerful geospatial insights. Designed and developed by [Mtaa Wetu](https://github.com/john-ngugi/mtaawetuapp), AGDA leverages modern spatial data science techniques combined with large language models (LLMs) to make geospatial data exploration and interpretation more interactive, more accessible, and more insightful.

> 🔗 **Repository:** https://github.com/john-ngugi/AGDA

---

## 🚀 What is AGDA?

AGDA is an AI-enhanced geospatial analyst that runs in the browser and helps users analyze, visualize, and understand GIS data through an intuitive interface. It was created to lower the barrier to entry for working with complex spatial datasets — by combining:

- **Streamlit** for interactive user interfaces,
- **GeoPandas & Folium** for GIS processing and mapping,
- **DeepSeek LLM** for intelligent natural language querying and interpretation.

AGDA is your spatial data co-pilot — able to read shapefiles, GeoJSON, and CSVs with geocoordinates, extract relevant metadata, generate choropleth and categorical maps, run basic statistics, and even explain the dataset structure in plain language.

---

## ✨ Features

### ✅ Upload and Explore Geospatial Data
- Supports `.geojson`, `.shp`, and `.csv` (with lat/lon)
- Automatically reprojects datasets to WGS84 (EPSG:4326)
- Displays number of features, geometry type, coordinate reference system (CRS), and attribute schema

### 🧠 AI Integration via DeepSeek
- LLM-powered agent that explains your dataset, summarizes key statistics, and answers spatial questions
- Example: "What is the most common land use category?" or "Show areas with population density over 1000"

### 🗺️ Interactive Map Visualizations
- Auto-centered Folium maps with intuitive zooming
- Attribute-based visualizations:
  - Choropleth maps for numeric fields
  - Color-coded maps for categorical fields
- Custom tooltips and hover interactivity
- Manual or dynamic legends

### 📊 Attribute Table & Statistics
- Explore raw attribute data in tabular form
- Dynamic column selection for map rendering
- Calculates value distributions, data types, and missing data

### 🧩 Modular and Extensible
- Modular Python codebase
- Easy to extend with new analysis tools, model backends, or visual styles

---

## 🛠️ Built With

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web interface and UI interactions |
| [GeoPandas](https://geopandas.org) | Geospatial data handling |
| [Folium](https://python-visualization.github.io/folium/) | Map rendering |
| [DeepSeek LLM](https://deepseek.com) | Language model for interpreting and summarizing spatial data |
| [Shapely, Pyproj, Pandas, NumPy](https://pypi.org) | Spatial operations, projections, and data manipulation |

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/john-ngugi/AGDA.git
cd AGDA

pip install -r requirements.txt

streamlit run app.py
```
AGDA/
├── app.py               # Main Streamlit app
├── agent/               # DeepSeek LLM integration
│   └── deepseek_agent.py
├── utils/               # Helper functions (map styling, file uploads, analysis)
│   └── map_utils.py
├── assets/              # Logo, icons, static files
├── requirements.txt
└── README.md


## 💡 Use Cases

- **Urban Planning**: Quickly identify underserved areas, visualize infrastructure gaps, and compare neighborhoods.
- **Community Mapping**: Allow citizens to upload and explore local spatial data without requiring GIS expertise.
- **Academic Research**: Visualize census or environmental datasets interactively.
- **Data Journalism**: Use AGDA to generate map-based stories from open spatial data.

---

## 🧠 Future Roadmap

| Version | Features |
|---------|----------|
| **v1.1** | CSV coordinate detection, multi-layer support, enhanced popups |
| **v1.2** | Exportable maps and reports (PDF/HTML), choropleth quantiles |
| **v2.0** | AI-driven insights dashboard, temporal data handling, multi-LLM support |

---

## 👥 Credits

AGDA is developed and maintained by the team at **Mtaa Wetu**, with ❤️ for the open data and community mapping movement.

- **Lead Developer**: John Ngugi  
- **Contact**: [admin@mtaawetu.com](mailto:admin@mtaawetu.com)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

> _"Let data speak, let communities thrive."_  
> — **AGDA**, your neighborhood data agent. 👋🌍

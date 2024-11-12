import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from datasets import load_dataset

# Load the dataset
dataset = load_dataset(DATASET_ID)['train']

# Extract data
data = {
    'x': [item['x'] for item in dataset],
    'y': [item['y'] for item in dataset],
    'label': [f"ID: {item['id']}" for item in dataset],
    'image': [item['image_url'] for item in dataset],
}

source = ColumnDataSource(data=data)


# Create the figure
p = figure(title="Image Similarity Data Visualization", tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,save,reset,hover", active_scroll="wheel_zoom",
           width=1500, height=1000, tooltips="""
    <div>
        <div><strong>@label</strong></div>
        <div><img src="@image" ></div>
    </div>
""")
p.circle('x', 'y', size=9, source=source)  # Apply the color mapper


st.set_page_config(layout='wide')

st.markdown("""
# Image Similarity Data Visualization

This visualization was created with [ImSimVis](https://github.com/TonyAssi/ImSimVis).

Images can be previewed on hover. Images position are based on similarity, images close to each other look similar. The colors represent the best seller index. 0 is best seller. [Dataset](DATASET_URL).
""")

#st.html('<br><br><br><br>')

# Display the Bokeh figure in Streamlit
st.bokeh_chart(p,use_container_width=True)



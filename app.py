import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io

# Load the pre-trained model
model = tf.keras.models.load_model('dep_model.h5')

# Check model input shape
input_shape = model.input_shape[1:3]  # Assuming the model input shape is (None, height, width, channels)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).resize(input_shape)  # Resize to model's expected input shape
    img_array = np.array(img) / 255.0  # Normalize to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Fire Detection"), className="text-center my-4")
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(
            id='uploader',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='output-image'), className="text-center")
    ]),
    dbc.Row([
        dbc.Col(html.H2(id='prediction'), className="text-center my-4")
    ])
])

# Define the callback for image upload and prediction
@app.callback(
    [Output('output-image', 'children'), Output('prediction', 'children')],
    [Input('uploader', 'contents')]
)
def update_output(contents):
    if contents is not None:
        # Parse the uploaded file content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Preprocess the image
        img_array = preprocess_image(decoded)
        
        # Predict using the model
        prediction = model.predict(img_array)
        
        # Determine if the image is fire or not
        is_fire = prediction[0, 0] > 0.5  # Assuming sigmoid activation in the last layer

        # Display the uploaded image
        image_element = html.Img(src=contents, style={'width': '50%'})

        # Display the prediction result
        prediction_text = "Fire Detected" if is_fire else "No Fire Detected"

        return image_element, prediction_text

    return None, ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

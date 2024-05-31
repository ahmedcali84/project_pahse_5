# Fire Detection Image Classification Project

This project focuses on building a deep learning model for fire detection in images using Convolutional Neural Networks (CNNs). The model is trained to classify images as either containing fire or not containing fire.

## Project Structure
- **`Overview.ipynb`**: Project overview and summary.
- **`preparation.ipynb`**: Details about data preparation.
- **`index.ipynb`**: Image Classification model training.
- **`app.py`**: Dash web application for uploading and classifying images.
- **`dep_model.h5`**: Pre-trained deep learning model for image classification.
- **`requirements.txt`**: List of Python packages required to run the application.

## Getting Started

### Prerequisites

- Python 3.
- Install required packages using `pip install -r requirements.txt`

### Running the Application

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using `pip install -r requirements.txt`.
4. Run the application using `python app.py`.
5. Access the web application at `http://127.0.0.1:8050/` in your browser.

## Usage

1. Drag and drop an image containing fire or a non-fire image onto the web application.
2. The application will classify the uploaded image as either "Fire Detected" or "No Fire Detected" and display the prediction along with the uploaded image.

## Model Details

The deep learning model used in this project is a CNN with the following architecture:

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2D) (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 28, 28, 64)        32832     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2D) (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 12, 12, 128)       73856     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2D) (None, 6, 6, 128)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 256)         295168    
_________________________________________________________________
max_pooling2d_7 (MaxPooling2D) (None, 2, 2, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 64)                65600     
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 65        
=================================================================
Total params: 468,417
Trainable params: 468,417
Non-trainable params: 0
_________________________________________________________________

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the creators of Dash, TensorFlow, and other libraries used in this project.

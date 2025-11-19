# ♻️ Waste Classification System
This project implements a Waste Classification System using a Machine learning model trained with TensorFlow and Keras, and deployed as a web application using Streamlit. The system can classify an uploaded image of waste into one of six categories: cardboard, glass, metal, paper, plastic, or trash.

# ⚙️ Installation and Setup

Required Python (3.x recommended)

### 1. Clone the Repository
- Open your terminal or command prompt and clone the project repository:

```
git clone https://github.com/m-santhosh-15/Waste-Classification.git
```

### 2. Install Dependencies

- Install all required Python libraries using the requirements.txt file.

```
pip install -r requirements.txt
```
The required libraries are streamlit, tensorflow, numpy, and pillow.

### 3. Configure Dataset Path
- The training script (train.py) requires a specific path to your dataset. You must update the DATASET_PATH variable in train.py to point to the location of your structured waste image dataset:

In train.py:

Python
```
#CHANGE THE PATH TO YOUR LOCAL DATASET LOCATION
DATASET_PATH =r"PATH"
```
(Note: The dataset folder should contain subdirectories, where each subdirectory name corresponds to a waste class, like cardboard, glass, metal, etc.).

### 4. Training and Model Generation
- Once the path is configured, run the training script to generate the model and the labels.txt file:

```
python train.py
```
This step will:

- Train the MobileNetV2-based model.

- Save the model as model/waste_model.h5.

- Generate the labels.txt file containing the classes (cardboard, glass, metal, paper, plastic, trash).

### 5. Run the Streamlit Application
- Finally, start the web application:
```
streamlit run app.py
```
The application will launch in your default web browser, allowing you to upload images for classification.

# OUTPUT
- https://santhosh-15-waste-classifier.hf.space/
##

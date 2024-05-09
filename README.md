# Budget Analyse

This project provides a machine learning solution for analyzing and predicting budget-related data, using a RandomForest algorithm. It is structured into three main components: model training, a Flask API for predictions, and a Streamlit frontend for user interaction.

## Overview

The repository consists of three main Python scripts:

- `model.py`: This script is responsible for training the machine learning model. It processes input data, trains a RandomForest model, and saves the model and its scaler to disk for later use by the API.
- `api.py`: Implements a Flask API that serves as the backend for the project. This API loads the trained model and uses it to make predictions based on incoming requests.
- `frontend.py`: A Streamlit application that provides a user-friendly web interface. Users can input data directly into the app, which communicates with the Flask API to display predictions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv (optional, for creating a virtual environment)

### Installation

1. Clone the repository and navigate to the correct folder:
   ```bash
   git clone https://github.com/Ismat-Samadov/Budget_Analyse.git
   cd Budget_Analyse/1.0.1
   ```

2. (Optional) Setup a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Model Training**:
   Run `model.py` to train the model and save it along with its scaler:
   ```bash
   python model.py
   ```

2. **Start the Flask API**:
   Ensure the model files are in place, then start the API with:
   ```bash
   python api.py
   ```
   The API listens for prediction requests and responds with predicted values.

3. **Launch the Streamlit Frontend**:
   With the Flask API running, start the Streamlit app:
   ```bash
   streamlit run frontend.py
   ```
   This opens a web interface where users can input data and receive predictions.

## Contributing

Feel free to fork the repository and submit pull requests. For substantial changes, please open an issue first to discuss what you would like to change. Make sure to update or add tests as appropriate.

## Contact

Ismat Samadov - [ismetsemedov@gmail.com](mailto:ismetsemedov@gmail.com)

Project Link: [https://github.com/Ismat-Samadov/Budget_Analyse](https://github.com/Ismat-Samadov/Budget_Analyse)
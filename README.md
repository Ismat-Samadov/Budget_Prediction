# Budget Analyse

This project provides a machine learning solution for analyzing and predicting budget-related data, using a RandomForest algorithm. It is designed to be interactive and user-friendly, encompassing model training, a Flask API for predictions, and a Streamlit frontend for user interaction.

## Overview

The repository is structured into three primary Python scripts:

- `model.py`: Handles the training of the machine learning model. It processes input data, trains a RandomForest model, and saves both the model and its scaler to disk.
- `api.py`: Implements a Flask API that serves predictions. This API loads the trained model from disk and uses it to predict based on incoming requests.
- `frontend.py`: A Streamlit application that provides a graphical interface. Users can input data directly, which is then sent to the Flask API to retrieve predictions.

## Further Reading

For more detailed insights into the project's architecture, development process, and a step-by-step guide, refer to our article on Medium:
- [Building a Budget Analysis Tool with Machine Learning and Python](https://ismatsamadov.medium.com/building-a-budget-analysis-tool-with-machine-learning-and-python-77954b2ec7a9)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv (optional for setting up a virtual environment)

### Installation

1. Clone the repository and navigate to the 1.0.2 folder:
   ```bash
   git clone https://github.com/Ismat-Samadov/Budget_Analyse.git
   cd Budget_Analyse/1.0.2
   ```

2. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Model Training**:
   Execute `model.py` to train the model and save it, along with its scaler:
   ```bash
   python model.py
   ```

2. **Start the Flask API**:
   Ensure the model files are available, then launch the API with:
   ```bash
   python api.py
   ```
   The API will listen for and respond to prediction requests.

3. **Launch the Streamlit Frontend**:
   With the API running, initiate the Streamlit app:
   ```bash
   streamlit run frontend.py
   ```
   This opens a web interface where users can input data and receive predictions in real-time.

### Live Application

Access the live application at: [Budget Analyse on Streamlit](https://budgett.streamlit.app/). This provides an interactive platform to explore the functionalities of the tool.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any changes. For major modifications, please open an issue first to discuss what you would like to change. Ensure you update or add appropriate tests as necessary.

## Contact

Ismat Samadov - [ismetsemedov@gmail.com](mailto:ismetsemedov@gmail.com)

Project Link: [https://github.com/Ismat-Samadov/Budget_Analyse](https://github.com/Ismat-Samadov/Budget_Analyse)
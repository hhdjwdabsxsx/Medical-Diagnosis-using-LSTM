# Medical Diagnosis with LSTM

## Project Overview
This project demonstrates the use of Long Short-Term Memory (LSTM) neural networks for medical diagnosis. The model predicts diseases and medications based on patient symptoms using deep learning techniques. The implementation leverages TensorFlow for building and training the model and includes the following components:

- **Patient Symptoms:** Textual descriptions of patients' symptoms.
- **Diagnoses:** Confirmed diseases for each patient.
- **Medications:** Prescribed treatments or medications.

The goal is to utilize sequence modeling to accurately predict medical conditions and associated prescriptions.

## Features
- **Data Tokenization and Preprocessing:** Converting text descriptions into numerical sequences for model processing.
- **LSTM-based Neural Network:** A sequence model designed to process patient symptom descriptions.
- **Dual Output Layers:** Simultaneously predicts disease diagnoses and medications.

## Technologies Used
- **TensorFlow**: For deep learning model architecture and training.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For label encoding.

## Dataset
The dataset includes:
- Text descriptions of patient symptoms.
- Disease diagnoses.
- Medications prescribed.

Ensure the dataset (`medical_data.csv`) is placed in the project directory. Example data structure:

| Patient Symptoms          | Diagnosis         | Medication      |
|---------------------------|-------------------|-----------------|
| Persistent cough, fever  | Bronchitis       | Amoxicillin     |
| Chest pain, short breath | Heart Disease    | Nitroglycerin   |

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/medical-diagnosis-lstm.git
   ```
2. Navigate to the project directory:
   ```bash
   cd medical-diagnosis-lstm
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Load the Dataset:**
   Ensure the dataset file `medical_data.csv` is in the root directory of the project.

2. **Run the Notebook:**
   Open and execute the `Medical_Diagnosis_using_LSTM.ipynb` file in Jupyter Notebook or any other compatible environment.

3. **Training the Model:**
   The notebook includes code for:
   - Data preprocessing (tokenization, padding, and encoding).
   - Model building and training.
   - Evaluation of predictions.

## Key Code Snippets
### Importing Libraries
```python
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
```

### Loading the Dataset
```python
data = pd.read_csv('medical_data.csv')
data.head()
```

### Building the LSTM Model
```python
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
lstm_layer = LSTM(128)(embedding_layer)
dense_disease = Dense(num_diseases, activation='softmax')(lstm_layer)
dense_medication = Dense(num_medications, activation='softmax')(lstm_layer)
model = Model(inputs=input_layer, outputs=[dense_disease, dense_medication])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Results
After training, the model predicts the disease and medication with reasonable accuracy. Evaluation metrics such as loss and accuracy are recorded for analysis.

## Future Work
- Enhance dataset diversity for broader applicability.
- Integrate more complex preprocessing pipelines for text data.
- Experiment with advanced deep learning architectures such as transformers.

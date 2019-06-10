# Systemic Feature Prediction in ROP

Networks to predict systemic features (gender, gestational age, postmenstrual age) of infants, especially made for infants with the possibility of plus disease in ROP

**Requirements**: 
- Python 3.6.x
- PyTorch 1.0.0
- CUDA 9.0
- GPU support

**How to Run**:
Run via command line (preferably in a docker with above requirements) using main.py file

`python main.py [action] [data_directory] [csv_file] [model_path]`

Actions: prepare, train, eval

Data Directory: contains all images for train and/or test, csv file must point to images in this directory

CSV file: csv file with columns of image name, plus disease classification, and feature (gender, gestational age in weeks, raw postmenstrual age, respectively)

Model Path: (path and name of location to save model) OR (path and name of trained model)

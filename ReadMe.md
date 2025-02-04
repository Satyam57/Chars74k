# Character Recognition using CHARS74K Dataset

This project demonstrates character recognition using the CHARS74K dataset from Kaggle. The project involves converting images into CSV files and using a Random Forest classifier for training and prediction.

## Dataset
The CHARS74K dataset contains images of characters from different languages, including English and Kannada. You can download the dataset from Kaggle [here](https://www.kaggle.com/).

## Project Structure
- `chars74k.py`: A script to convert images into `chars74k_train.csv` and `chars74k_test.csv`.
- `chars74k_pred.py`: A script to train and predict using the Random Forest classifier with the CSV files.
- `README.md`: This file.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- numpy
- OpenCV

Install the required packages using:
```sh
pip install pandas scikit-learn numpy opencv-python

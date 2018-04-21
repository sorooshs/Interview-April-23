from app.models import TrainerEstimator
from app.data_loader import DataLoader
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

market_id_list = range(1, 6)


def main():
    prediction_df = DataLoader().load_prediction_data()
    pd.DataFrame(TrainerEstimator()).predict(prediction_df).to_csv('output.csv', index=False)

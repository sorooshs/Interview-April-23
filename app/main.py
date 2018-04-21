from app.models import TrainerEstimator
from app.data_loader import DataLoader
import pandas as pd


def main():
    prediction_df = DataLoader().load_prediction_data()
    pd.DataFrame(TrainerEstimator(market_id=3, store_primary_category='american')
                 .predict(prediction_df)).to_csv('output.csv')

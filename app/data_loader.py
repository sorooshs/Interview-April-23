
import pandas as pd
import datetime


class DataLoader(object):

    train_filename = 'Data_Science/historical_data.csv'
    test_filename = 'Data_Science/data_to_predict.json'

    features_col_name = ['market_id',
                         'order_protocol',
                         'total_items',
                         'subtotal',
                         'num_distinct_items',
                         'min_item_price',
                         'max_item_price',
                         'total_onshift_dashers',
                         'total_busy_dashers',
                         'total_outstanding_orders',
                         'estimated_order_place_duration',
                         'estimated_store_to_consumer_driving_duration']

    def load_train_data(self):
        input_df = pd.read_csv(self.train_filename)

        # Convert to datetime
        input_df.created_at = pd.to_datetime(input_df.created_at)
        input_df.actual_delivery_time = pd.to_datetime(input_df.actual_delivery_time)

        time_split = input_df['created_at'] < datetime.date(2015, 2, 17)
        train_df = input_df[time_split]
        test_df = input_df[~time_split]

        test_df.dropna(subset=['actual_delivery_time'], inplace = True)
        test_df.dropna(subset=['created_at'], inplace = True)

        # print('''len train: {},
        #     len test: {}'''.format(len(train_df), len(test_df)))

        return train_df, test_df

    def load_prediction_data(self):
        df = pd.read_json(open(self.test_filename))
        for col in self.features_col_name:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

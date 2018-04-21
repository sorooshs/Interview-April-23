from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from app.data_loader import DataLoader
import logging


logger = logging.getLogger(__name__)


def train_data_feature_generation(df):
    """
    Clean and generate new features for train data
    Args:
        df: train data dataframe

    Return:
        Cleaned train data dataframe
    """

    # Remove intances without label and convert time to datetime
    # Drop NAs in actual_delivery_time column
    df.dropna(subset=['actual_delivery_time'], inplace=True)
    df.dropna(subset=['created_at'], inplace=True)

    # Convert to datetime
    df.created_at = pd.to_datetime(df.created_at)
    df.actual_delivery_time = pd.to_datetime(df.actual_delivery_time)

    df['total_delivery_duration_seconds'] = (df.actual_delivery_time - df.created_at).dt.total_seconds()

    # Best Store primary category
    best_store_primary_category_df = df \
        .groupby(['store_primary_category', 'store_id']) \
        .agg({'store_primary_category': 'count'}) \
        .rename(columns={'store_primary_category': 'store_primary_category_count'}) \
        .reset_index() \
        .sort_values('store_primary_category_count', ascending=False) \
        .drop_duplicates('store_id') \
        .set_index('store_id')

    # Add new features
    best_store_primary_category_dict = best_store_primary_category_df['store_primary_category'].to_dict()
    df['store_primary_category'] = df['store_id'].map(best_store_primary_category_dict).fillna(value='other')

    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['hour_of_order'] = df['created_at'].dt.hour
    df['minutes_of_order'] = df['created_at'].dt.minute
    df['store_id_int'] = df['store_id'] \
        .str.slice(0, 8) \
        .apply(lambda x: int(x, 16))
    df['total_onshift_available'] = df['total_onshift_dashers'] - df['total_busy_dashers']

    # Replace negative with zero
    num = df._get_numeric_data()
    num[num < 0] = 0

    df = df.fillna(0)

    return df


def test_data_feature_generation(df):

    """
    Generate new features for test dataframe
    Args:
        df: test data dataframe

    Return:
        test data dataframe with new features
    """

    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['hour_of_order'] = df['created_at'].dt.hour
    df['minutes_of_order'] = df['created_at'].dt.minute
    df['store_id_int'] = df['store_id'] \
        .str.slice(0, 8) \
        .apply(lambda x: int(x, 16))

    df['total_onshift_available'] = df['total_onshift_dashers'] - df['total_busy_dashers']

    # Replace negative with zero
    num = df._get_numeric_data()
    num[num < 0] = 0

    return df


class TrainerEstimator(object):

    label_col_name = 'total_delivery_duration_seconds'
    features_col_name = ['market_id',
                         'order_protocol',
                         'total_items',
                         'subtotal',
                         'num_distinct_items',
                         'min_item_price',
                         'max_item_price',
                         'total_onshift_dashers',
                         'total_busy_dashers',
                         'total_onshift_available',
                         'total_outstanding_orders',
                         'estimated_order_place_duration',
                         'estimated_store_to_consumer_driving_duration',
                         'day_of_week',
                         'hour_of_order',
                         'minutes_of_order',
                         'store_id_int']

    def __init__(self):
        train_df, validation_df = DataLoader().load_train_data()
        # self.model, self.train_median_dict = self.create_delivery_models(train_df)

        self.model_dict = {}
        for (category, id) in self.get_market_id_categories(train_df):
            self.model_dict[(category, id)] = self.create_delivery_models(train_df, category, id)

    def get_market_id_categories(self, df):
        g = df.groupby(['store_primary_category', 'market_id']).count().add_suffix('_Count').reset_index()
        return zip(g['store_primary_category'], g['market_id'])

    def create_delivery_models(self, train_df, store_primary_category, market_id):
        market_id = int(market_id)
        print("create_delivery_models({}, {})".format(store_primary_category, market_id))

        market_id_filter = train_df.market_id == market_id
        store_primary_filter = train_df.store_primary_category == store_primary_category

        # If there was enough data to create a looser model
        if len(train_df[market_id_filter & store_primary_filter]) < 50:
            train_df = train_df[market_id_filter | store_primary_filter]
        else:
            train_df = train_df[market_id_filter & store_primary_filter]

        train_df = train_data_feature_generation(train_df)

        train_median_dict = train_df.dropna().median().to_dict()
        train_df = train_df.fillna(train_median_dict)

        mean = train_df['total_delivery_duration_seconds'].mean()
        std = train_df['total_delivery_duration_seconds'].std()
        outlier_high_range = mean + 2 * std

        not_outliers_filter = train_df['total_delivery_duration_seconds'] < outlier_high_range
        train_df = train_df[not_outliers_filter]

        model = RandomForestRegressor(n_estimators=100,
                                      n_jobs=-1,
                                      random_state=3,
                                      max_depth=7)

        X_train = train_df[self.features_col_name].values
        y_train = train_df[self.label_col_name].values

        model.fit(X_train, y_train)

        return model, train_median_dict

    def predict(self, prediction_df):
        prediction_df = test_data_feature_generation(prediction_df)
        prediction_df = prediction_df.fillna(self.train_median_dict)
        # p = self.model.predict(prediction_df[self.features_col_name].values)
        # prediction_df['predicted_delivery_seconds'] = p
        #
        # return prediction_df[['delivery_id', 'predicted_delivery_seconds']]

        out = list()
        for index, row in prediction_df[self.features_col_name].iterrows():
            model, train_median_dict = self.model_dict[(row['store_primary_category'], row['market_id'])]

            p = model.predict(row)
            row['predicted_delivery_seconds'] = p
            out.append(row[['delivery_id', 'predicted_delivery_seconds']])

        return pd.DataFrame(out)

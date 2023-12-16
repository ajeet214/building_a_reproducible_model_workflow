
import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """
    Test whether the columns in the dataset match the expected column names.

    Args:
        data (pd.DataFrame): The dataset to be tested.
    """

    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    # Enforce the same order
    assert list(expected_columns) == list(data.columns.values)


def test_neighborhood_names(data):

    """
    Test whether the 'neighbourhood_group' column contains known neighborhood names.

    Args:
        data (pd.DataFrame): The dataset to be tested.
    """

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC.

    Args:
        data (pd.DataFrame): The dataset to be tested.
    """

    idx = data['longitude'].between(-74.25, -
                                    73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset.

    Args:
        data (pd.DataFrame): The dataset to be tested.
        ref_data (pd.DataFrame): The reference dataset for comparison.
        kl_threshold (float): The threshold for the KL divergence.
    """

    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    Test that the price range is within the expected boundaries.

    Args:
        data (pd.DataFrame): The dataset to be tested.
        min_price (float): The minimum expected price.
        max_price (float): The maximum expected price.
    """

    assert data['price'].between(min_price, max_price).all()


def test_row_count(data):
    """
    Test that the number of rows in the dataset is within the expected boundaries.

    Args:
        data (pd.DataFrame): The dataset to be tested.
    """

    assert 15000 < data.shape[0] < 100000

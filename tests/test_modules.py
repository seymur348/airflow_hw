import pandas as pd
import pytest

from modules.pipeline import create_features, filter_data, remove_outliers
from modules.validate import check_data


def make_df(n=60):
    return pd.DataFrame({
        'id': range(n),
        'year': [2010 + i % 10 for i in range(n)],
        'model': ['camry le'] * n,
        'price_category': ['low', 'medium', 'high'] * (n // 3),
    })


def test_filter_data_drops_service_columns():
    df = make_df().assign(url='u', price=1, lat=0.0)
    out = filter_data(df.assign(region='r', region_url='r', manufacturer='m', image_url='i',
                                description='d', posting_date='p', long=0.0))
    assert not {'id', 'url', 'price', 'lat', 'long'} & set(out.columns)
    assert {'year', 'model'} <= set(out.columns)


def test_remove_outliers_clips_year():
    df = pd.DataFrame({'year': [2010, 2011, 2012, 2013, 2014, 1900]})
    out = remove_outliers(df)
    assert out['year'].min() > 1900
    assert df['year'].min() == 1900  # исходная таблица не изменена


@pytest.mark.parametrize('year,expected', [(2020, 'new'), (2000, 'old'), (2010, 'average')])
def test_age_category(year, expected):
    out = create_features(pd.DataFrame({'year': [year], 'model': ['Camry LE']}))
    assert out.loc[0, 'age_category'] == expected
    assert out.loc[0, 'short_model'] == 'camry'


def test_check_data_ok():
    assert check_data(make_df())['rows'] == 60


def test_check_data_missing_column():
    with pytest.raises(ValueError, match='обязательных колонок'):
        check_data(make_df().drop(columns=['year']))


def test_check_data_too_few_rows():
    with pytest.raises(ValueError, match='мало строк'):
        check_data(make_df(6))


def test_check_data_target_missing():
    df = make_df()
    df.loc[0, 'price_category'] = None
    with pytest.raises(ValueError, match='целевой'):
        check_data(df)


def test_check_data_mostly_empty_column():
    df = make_df()
    df.loc[:40, 'model'] = None
    with pytest.raises(ValueError, match='пропусков'):
        check_data(df)

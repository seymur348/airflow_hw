"""Генератор СИНТЕТИЧЕСКИХ данных для проверки конвейера.

Данные случайные и нужны только для того, чтобы проект запускался с нуля.
Для реальных результатов положите настоящий data/train/homework.csv.

Запуск из корня проекта:  python scripts/make_sample_data.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('PROJECT_PATH', '.'))
rng = np.random.default_rng(42)

MAKERS = {  # производитель -> (модели, ценовой коэффициент)
    'toyota': (['camry le', 'corolla l', 'rav4 xle', 'tacoma sr5'], 1.1),
    'honda': (['civic lx', 'accord ex', 'cr-v ex'], 1.05),
    'ford': (['f-150 xlt', 'focus se', 'escape se', 'mustang gt'], 0.95),
    'chevrolet': (['silverado 1500', 'malibu lt', 'equinox lt'], 0.9),
    'bmw': (['328i', 'x5 xdrive', '530i'], 1.5),
    'nissan': (['altima s', 'rogue sv', 'sentra sv'], 0.85),
}
FUELS = ['gas', 'diesel', 'hybrid', 'other']
TRANSMISSIONS = ['automatic', 'manual', 'other']
STATUSES = ['clean', 'rebuilt', 'salvage']
STATES = ['ca', 'fl', 'ny', 'tx', 'wa']


def make_frame(n: int, start_id: int) -> pd.DataFrame:
    makers = rng.choice(list(MAKERS), n)
    models = [rng.choice(MAKERS[m][0]) for m in makers]
    coef = np.array([MAKERS[m][1] for m in makers])
    year = np.clip(rng.normal(2012, 6, n).round(), 1990, 2021).astype(int)
    odometer = np.clip(rng.normal(110000 - (year - 2000) * 1500, 40000, n), 500, 400000).round()
    price = (3000 + (year - 1990) * 750 - odometer * 0.04) * coef + rng.normal(0, 1500, n)
    price = np.clip(price, 500, None).round()
    ids = start_id + np.arange(n)
    df = pd.DataFrame({
        'id': ids,
        'url': [f'https://example.org/cars/{i}' for i in ids],
        'region': rng.choice(['sf bay area', 'miami', 'new york', 'dallas', 'seattle'], n),
        'region_url': 'https://example.org',
        'price': price,
        'year': year.astype(float),
        'manufacturer': makers,
        'model': models,
        'fuel': rng.choice(FUELS, n, p=[0.8, 0.08, 0.08, 0.04]),
        'odometer': odometer,
        'title_status': rng.choice(STATUSES, n, p=[0.85, 0.1, 0.05]),
        'transmission': rng.choice(TRANSMISSIONS, n, p=[0.8, 0.15, 0.05]),
        'image_url': 'https://example.org/img.jpg',
        'description': 'synthetic sample row',
        'state': rng.choice(STATES, n),
        'lat': rng.uniform(25, 48, n).round(4),
        'long': rng.uniform(-123, -73, n).round(4),
        'posting_date': '2021-05-04T12:00:00+0000',
    })
    # немного пропусков, как в реальных данных
    for col in ['odometer', 'fuel', 'transmission', 'model']:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return df


def main() -> None:
    train = make_frame(1000, 7_000_000_000)
    q1, q2 = train['price'].quantile([0.33, 0.66])
    train['price_category'] = np.where(train['price'] <= q1, 'low',
                                       np.where(train['price'] <= q2, 'medium', 'high'))
    (ROOT / 'data/train').mkdir(parents=True, exist_ok=True)
    train.to_csv(ROOT / 'data/train/homework.csv', index=False)

    test = make_frame(5, 7_400_000_000).drop(columns=[])
    (ROOT / 'data/test').mkdir(parents=True, exist_ok=True)
    for old in (ROOT / 'data/test').glob('*.json'):
        old.unlink()
    for _, row in test.iterrows():
        rec = {k: (None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v)) for k, v in row.items()}
        with open(ROOT / f'data/test/{int(rec["id"])}.json', 'w') as f:
            json.dump(rec, f, ensure_ascii=False)
    (ROOT / 'data/models').mkdir(parents=True, exist_ok=True)
    print('train:', train.shape, '| class balance:', train['price_category'].value_counts().to_dict())
    print('test json files:', len(test))


if __name__ == '__main__':
    main()

import logging
import os

import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')

REQUIRED_COLUMNS = ['id', 'year', 'model', 'price_category']
TARGET = 'price_category'
MAX_MISSING_SHARE = 0.5   # колонка, где пропусков больше, считается испорченной
MIN_ROWS = 50


def check_data(df: pd.DataFrame) -> dict:
    """Проверяет таблицу. Критичные проблемы -> ValueError, остальное -> warning."""
    errors, warnings = [], []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Нет обязательных колонок: {missing_cols}')

    if len(df) < MIN_ROWS:
        errors.append(f'Слишком мало строк: {len(df)} < {MIN_ROWS}')

    if df[TARGET].isna().any():
        errors.append(f'В целевой колонке {TARGET} есть пропуски: {int(df[TARGET].isna().sum())}')

    if df[TARGET].nunique() < 2:
        errors.append(f'В {TARGET} меньше двух классов')

    share = df.isna().mean()
    broken = share[share > MAX_MISSING_SHARE]
    if not broken.empty:
        errors.append(f'Слишком много пропусков (> {MAX_MISSING_SHARE:.0%}): {broken.round(2).to_dict()}')

    partial = share[(share > 0) & (share <= MAX_MISSING_SHARE)]
    if not partial.empty:
        warnings.append(f'Есть пропуски, они будут заполнены в пайплайне: {partial.round(3).to_dict()}')

    dups = int(df.duplicated(subset='id').sum())
    if dups:
        warnings.append(f'Дубликаты по id: {dups}')

    if errors:
        raise ValueError('Данные не прошли проверку: ' + '; '.join(errors))

    for w in warnings:
        logging.warning(w)
    return {'rows': len(df), 'columns': df.shape[1], 'warnings': warnings}


def validate_data() -> None:
    df = pd.read_csv(f'{path}/data/train/homework.csv')
    summary = check_data(df)
    logging.info(f'Проверка данных пройдена: строк {summary["rows"]}, колонок {summary["columns"]}, '
                 f'предупреждений {len(summary["warnings"])}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    validate_data()

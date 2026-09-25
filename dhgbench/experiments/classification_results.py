"""Read completed classification cells without mixing study protocols."""
import csv
import json
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from lib_models import _semi_methods_


CSV_METRIC_COLUMNS = ('tune_val', 'test_acc', 'test_std', 'val_acc')
CSV_DECIMAL_PLACES = 2


def format_csv_row(row):
    """Format presentation metrics without rounding the underlying record."""
    formatted = dict(row)
    quantum = Decimal(1).scaleb(-CSV_DECIMAL_PLACES)
    for column in CSV_METRIC_COLUMNS:
        value = row.get(column)
        if value is None or value == '':
            continue
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError(f'Nonfinite classification metric: {column}={value}')
        formatted[column] = format(
            number.quantize(quantum, rounding=ROUND_HALF_EVEN),
            f'.{CSV_DECIMAL_PLACES}f')
    return formatted


def collect(root, kind='selected'):
    root = Path(root)
    rows = []
    if kind == 'selected':
        for path in sorted(root.glob('*/*_L*.json')):
            rows.append(json.loads(path.read_text())['summary'])
    elif kind == 'evaluated':
        for path in sorted((root / 'evaluated').glob('*/*_L*.csv')):
            with path.open() as handle:
                rows.extend(csv.DictReader(handle))
    else:
        raise ValueError(f'Unknown result kind: {kind}')
    allowed = set(_semi_methods_)
    rows = [row for row in rows if row['method'] in allowed]
    if any(row.get('protocol') != 'main-depth-matched-hnhn-v1' for row in rows):
        raise ValueError('Run directory contains historical results; use a directory from the current main study')
    if not rows:
        raise ValueError(f'No completed {kind} cells found in {root}')
    keys = [(row['dataset'], row['method'], int(row['depth'])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate cells: use a directory containing one experiment')
    return rows


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fields differ across saved, tuned, and matched protocols.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(format_csv_row(row) for row in rows)
    temporary.replace(path)

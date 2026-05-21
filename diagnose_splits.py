import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from modelling_tuning import load_preprocessed_data

if __name__ == '__main__':
    force_agg = os.environ.get('FORCE_AGG')
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(force_aggregation=force_agg)

    print('Shapes:')
    print('X_train', X_train.shape)
    print('X_val  ', X_val.shape)
    print('X_test ', X_test.shape)
    print('\nTarget statistics:')
    def stats(a):
        a = np.asarray(a, dtype=float)
        return {'n': len(a), 'mean': float(np.nanmean(a)), 'std': float(np.nanstd(a)), 'min': float(np.nanmin(a)), 'max': float(np.nanmax(a))}

    print('y_train', stats(y_train))
    print('y_val  ', stats(y_val))
    print('y_test ', stats(y_test))

    out_dir = Path('Membangun_model/artifacts')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.hist(y_train, bins=40, alpha=0.5, label='train')
    plt.hist(y_val, bins=40, alpha=0.5, label='val')
    plt.hist(y_test, bins=40, alpha=0.5, label='test')
    plt.legend()
    plt.title('Target distribution by split')
    p = out_dir / 'target_split_hist.png'
    plt.tight_layout()
    plt.savefig(p)
    print('\nSaved histogram to:', p)

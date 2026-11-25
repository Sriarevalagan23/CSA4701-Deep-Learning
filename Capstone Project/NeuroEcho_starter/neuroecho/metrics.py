import numpy as np

def regression_metrics(reference, prediction):
    ref = np.asarray(reference).reshape(-1)
    pred = np.asarray(prediction).reshape(-1)
    mse = float(np.mean((ref - pred) ** 2))
    mae = float(np.mean(np.abs(ref - pred)))
    corr = float(np.corrcoef(ref, pred)[0, 1]) if np.std(ref) > 1e-8 and np.std(pred) > 1e-8 else 0.0
    return {"MSE": mse, "MAE": mae, "Pearson correlation": corr}

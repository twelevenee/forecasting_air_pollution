import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalTransform(df, x):
    df[x] = 1/np.sqrt(2*np.pi*df[x].std()) * np.exp(- np.power((df[x] - df[x].mean()) /(df[x].std()), 2) / 2)
    return df

original_targets = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']

def addLagFeatures(df, lags, targets):
    for tgt in targets:
        for lag in lags:
            df[f'{tgt}_lag{lag}'] = df[tgt].shift(lag)
    return df

def addMovingAverages(df, windows, targets):
    for tgt in targets:
        for w in windows:
            df[f'{tgt}_MA{w}'] = df[tgt].rolling(window=w, min_periods=1).mean()
    return df

def LagMACorrelations(df, targets):
    rows = []
    for tgt in targets:
        # Find MA features for this target
        ma_cols = [c for c in df.columns if c.startswith(f'{tgt}_MA')]
        for col in ma_cols:
            corr = df[[tgt, col]].corr().iloc[0,1]
            rows.append((tgt, col, corr))
        
        # Find lag features for this target
        lag_cols = [c for c in df.columns if c.startswith(f'{tgt}_lag')]
        for col in lag_cols:
            corr = df[[tgt, col]].corr().iloc[0,1]
            rows.append((tgt, col, corr))
    
    return pd.DataFrame(rows, columns=['target','feature','correlation'])

def plot_lag_scatter(df, targets, lags):
    """
    Creates scatter plots of each target vs each of its lag features.
    Layout: one row per target, one column per lag.
    """

    n_targets = len(targets)
    n_lags = len(lags)

    fig, axes = plt.subplots(n_targets, n_lags, figsize=(5*n_lags, 4*n_targets), squeeze=False)

    for i, tgt in enumerate(targets):
        for j, lag in enumerate(lags):
            ax = axes[i][j]

            lag_col = f"{tgt}_lag{lag}"
            if lag_col not in df.columns:
                ax.set_title(f"{lag_col} missing")
                ax.axis("off")
                continue

            # Scatter plot
            ax.scatter(df[lag_col], df[tgt], alpha=0.4, s=10)
            ax.set_xlabel(lag_col)
            ax.set_ylabel(tgt)
            ax.set_title(f"{tgt} vs {lag_col}")

    plt.tight_layout()
    plt.show()
    return 

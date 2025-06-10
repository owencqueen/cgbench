import numpy as np

def precision_recall_at_k(
        y_true: np.ndarray,
        y_pred_or_scores: np.ndarray,
        from_scores: bool = False,
):
    """
    Precision@k and recall@k for *one* multi-label example.

    Parameters
    ----------
    y_true              : (C,)   binary array – ground-truth labels
    y_pred_or_scores    : (C,)   OR (k, C)  array
                           * binary (0/1)     when from_scores=False
                           * real-valued      when from_scores=True
                           If 1-D, it is treated as the usual C-long vector.
                           If 2-D (k, C), each row is the model's i-th guess.
    k                   : int    number of predictions that define “@k”
    from_scores         : bool   treat `y_pred_or_scores` as raw scores/logits
                                 (top-1 per row is taken) instead of binary.

    Returns
    -------
    precision_k, recall_k : floats
    """
    k = y_pred_or_scores.shape[0]
    # ---- sanity checks --------------------------------------------------
    y_true = np.asarray(y_true, dtype=int)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1-D (C,)")

    y_pred = np.asarray(y_pred_or_scores)

    if y_pred.ndim == 1:                       # ➊ Classic (C,) case
        y_pred = y_pred[np.newaxis, :]         # → shape (1, C)

    if y_pred.ndim != 2 or y_pred.shape[1] != y_true.size:
        raise ValueError("y_pred_or_scores must be (C,) or (k, C)")

    k_rows = y_pred.shape[0]
    if k_rows != k:
        raise ValueError(f"First dimension must equal k={k}")

    # ---- convert rows → one-hot binaries if we’re given scores ----------
    if from_scores:
        # argmax per row → one-hot
        idx = np.argmax(y_pred, axis=1)
        y_pred_bin = np.zeros_like(y_pred, dtype=int)
        y_pred_bin[np.arange(k_rows), idx] = 1
        y_pred = y_pred_bin
    else:
        # verify binary
        if not np.isin(y_pred, [0, 1]).all():
            raise ValueError("y_pred must be binary when from_scores=False")

    # ---- obtain the set of *unique* predicted class indices -------------
    pred_indices = np.where(y_pred.sum(axis=0) > 0)[0]   # union over rows
    hits = y_true[pred_indices].sum()                    # true positives
    pos_true = y_true.sum()

    # if y_true.sum() > 1:
    #     import ipdb; ipdb.set_trace()

    precision_k = y_pred[:,y_true.astype(bool)].sum() / k

    #precision_k = hits / k if k else 0.0
    recall_k    = hits / pos_true if pos_true else 0.0

    return precision_k, recall_k

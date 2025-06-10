import argparse
import pandas as pd

import numpy as np
from scipy.stats import beta, binom

confusion_matrix = [14, 2, 6, 12]

def corrected_rate_MCBootstrap(tp, fp, fn, tn,
                               n_pos_obs,            # F says "correct" on G
                               n_neg_obs,            # F says "incorrect"
                               B=10_000,             # boot iterations
                               alpha=0.05):
    N = n_pos_obs + n_neg_obs

    # posterior for Se (Beta)  and Sp (Beta)  using flat priors
    a_se, b_se = tp + 1, fn + 1
    a_sp, b_sp = tn + 1, fp + 1

    # observed positive-rate posterior on G’s sample
    a_p,  b_p  = n_pos_obs + 1, n_neg_obs + 1

    draws = []
    for _ in range(B):
        Se  = beta.rvs(a_se, b_se)
        Sp  = beta.rvs(a_sp, b_sp)
        p_o = beta.rvs(a_p, b_p)             # binomial sampling uncertainty

        denom = Se + Sp - 1
        if denom <= 0:                       # impossible channel; skip draw
            continue

        p_true = (p_o + Sp - 1) / denom      # Rogan–Gladen correction
        p_true = np.clip(p_true, 0, 1)       # keep inside (0,1)
        draws.append(p_true)

    lo, hi = np.percentile(draws, [100*alpha/2, 100*(1-alpha/2)])
    return np.mean(draws), (lo, hi)


def main(args):
    df = pd.read_csv(args.path)

    # Get the best of k judgements:
    n_pos_obs = df.judgement.sum()
    n_neg_obs = len(df) - n_pos_obs

    pct_correct, (lo, hi) = corrected_rate_MCBootstrap(
        tp=confusion_matrix[0],
        fp=confusion_matrix[1],
        fn=confusion_matrix[2],
        tn=confusion_matrix[3],
        n_pos_obs=n_pos_obs,
        n_neg_obs=n_neg_obs)

    print(f"Pct correct: {pct_correct} ({lo}, {hi})")
    std_str = r"{" + f"{df.judgement.std() / np.sqrt(df.shape[0]):.3f}" + r"}"
    print(f"Pct correct (no correction): {n_pos_obs / len(df):.3f}\\std{std_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
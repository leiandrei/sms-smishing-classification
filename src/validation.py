from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

TARGET_NAMES = ['Smishing', 'Legitimate']        

SIGNAL_COLS = {                                    
    'has_url', 'has_money_mention', 'has_phone_in_text', 'sender_is_alphanumeric',
    'exclamation_count', 'text_count', 'word_count', 'punctuation_count',
    'uppercase_ratio',
}


def build_feature_names(vectorizer, binary_cols=(), numericals=()) -> np.ndarray:
    """Column names for hstack([tfidf, binary_cols, numericals]), same order."""
    return np.array(list(vectorizer.get_feature_names_out())
                    + list(binary_cols) + list(numericals))


def _strip_prefix(name: str) -> str:
    return name.split('__', 1)[-1] if '__' in name else name


def resolve_model(model: BaseEstimator, feature_names=None):
    """Return (classifier, feature_names, preprocessor_or_None).

    * Pipeline  -> last step is the classifier; names come from the earlier
                   steps (ColumnTransformer prefixes like 'tfidf__' are removed).
    * Bare clf  -> `feature_names` is required.
    """
    if hasattr(model, 'steps'):
        clf, pre = model.steps[-1][1], model[:-1]
        if feature_names is None:
            try:
                feature_names = pre.get_feature_names_out()
            except Exception:
                vec = next((s for _, s in model.steps[:-1]
                            if hasattr(s, 'get_feature_names_out')), None)
                if vec is None:
                    raise ValueError("No vectorizer step found; pass feature_names.")
                feature_names = vec.get_feature_names_out()
    else:
        clf, pre = model, None
        if feature_names is None:
            raise ValueError("Bare classifier: pass feature_names "
                             "(see build_feature_names).")

    names = np.array([_strip_prefix(str(n)) for n in feature_names])
    width = getattr(clf, 'n_features_in_', None)
    if width is not None and width != len(names):
        raise ValueError(
            f"Model was fitted on {width} columns but {len(names)} feature names "
            "were given. If you stacked signal columns onto the TF-IDF matrix, "
            "use build_feature_names(vectorizer, binary_cols, numericals).")
    return clf, names, pre


def _kind(names: np.ndarray) -> np.ndarray:
    return np.where(np.isin(names, list(SIGNAL_COLS)), 'signal', 'word')


def _ngram_ok(names: np.ndarray, max_ngram) -> np.ndarray:
    if max_ngram is None:
        return np.ones(len(names), dtype=bool)
    return np.array([len(n.split()) <= max_ngram for n in names])


def _as_matrix(X, pre):
    """Raw text/DataFrame -> matrix via the pipeline's preprocessor; matrices pass through."""
    if pre is not None and not (sp.issparse(X) or isinstance(X, np.ndarray)):
        return pre.transform(X)
    return X


def _top_per_class(tbl: pd.DataFrame, by: str, top_n: int) -> pd.DataFrame:
    out = (tbl.sort_values(by, ascending=False)
              .groupby('class', sort=False).head(top_n))
    out = out.sort_values(['class', by], ascending=[True, False]).reset_index(drop=True)
    out.insert(1, 'rank', out.groupby('class').cumcount() + 1)
    return out

def term_stats(X, y, feature_names, class_names=TARGET_NAMES) -> pd.DataFrame:
    """Per feature: number of messages containing it in each class, its mean
    value in each class, and `lean` (the class where the mean is higher).

    Use the TRAINING matrix: `stats = term_stats(X_train, y_train, names)`.
    """
    X, y = sp.csr_matrix(X), np.asarray(y)
    cols = {'feature': np.asarray(feature_names)}
    for c, cname in enumerate(class_names):
        Xc = X[y == c]
        cols[f'docs_{cname}'] = np.asarray((Xc > 0).sum(axis=0)).ravel()
        cols[f'mean_{cname}'] = np.asarray(Xc.mean(axis=0)).ravel()
    stats = pd.DataFrame(cols)
    means = stats[[f'mean_{n}' for n in class_names]].to_numpy()
    stats['lean'] = np.where(means.max(1) == means.min(1), 'none',
                             np.array(class_names)[means.argmax(1)])
    return stats

def nb_word_table(model, class_names=TARGET_NAMES, feature_names=None,
                  stats: pd.DataFrame | None = None, top_n: int = 20,
                  min_docs: int = 3, kinds=('word',),
                  max_ngram: int | None = None) -> pd.DataFrame:
    """Top features per class by relative NB weight:

        score = feature_log_prob_[class] - mean(feature_log_prob_[other classes])

    Positive = pushes toward that class. For MultinomialNB this is a log-odds.
    For ComplementNB the weights are complement-based and rescaled, so use the
    score for RANKING within a model, not as a probability statement.

    `stats` (from term_stats) enables `min_docs`: a word must appear in at least
    that many messages of the class it is credited to, otherwise rare 2-message
    words with extreme weights dominate the list. `max_ngram=1` keeps single
    words only (bigrams/trigrams repeat the same evidence).
    """
    clf, names, _ = resolve_model(model, feature_names)
    if not hasattr(clf, 'feature_log_prob_'):
        raise TypeError("nb_word_table needs a fitted Naive Bayes classifier.")
    flp, k = clf.feature_log_prob_, len(class_names)

    base = pd.DataFrame({'feature': names, 'kind': _kind(names)})
    if stats is not None:
        base = base.merge(stats, on='feature', how='left')

    parts = []
    for c, cname in enumerate(class_names):
        others = [i for i in range(k) if i != c]
        t = base.copy()
        t['score'] = flp[c] - flp[others].mean(axis=0)
        t['class'] = cname
        if stats is not None:
            t = t[t[f'docs_{cname}'] >= min_docs]
        t = t[t['kind'].isin(kinds) & _ngram_ok(t['feature'].to_numpy(), max_ngram)]
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    return _top_per_class(out, 'score', top_n)

def rf_importance_table(model, class_names=TARGET_NAMES, feature_names=None,
                        stats: pd.DataFrame | None = None, top_n: int = 20,
                        min_docs: int = 3, kinds=('word',),
                        max_ngram: int | None = None) -> pd.DataFrame:
    """Impurity-based importance. It has no direction, so with `stats` each
    feature is filed under the class it is more common in (`lean`). Without
    `stats` everything goes under class '(all)'.

    Caveat: impurity importance favours continuous features (text_count,
    uppercase_ratio) over rare words. Pass kinds=('word','signal') to see them
    side by side, and confirm with rf_permutation_table().
    """
    clf, names, _ = resolve_model(model, feature_names)
    if not hasattr(clf, 'feature_importances_'):
        raise TypeError("rf_importance_table needs a fitted tree ensemble.")
    t = pd.DataFrame({'feature': names, 'kind': _kind(names),
                      'importance': clf.feature_importances_})
    if stats is not None:
        t = t.merge(stats, on='feature', how='left')
        t['class'] = t['lean']
        docs = t[[f'docs_{c}' for c in class_names]].sum(axis=1)
        t = t[docs >= min_docs]
    else:
        t['class'] = '(all)'
    t = t[t['kind'].isin(kinds) & _ngram_ok(t['feature'].to_numpy(), max_ngram)]
    return _top_per_class(t, 'importance', top_n)

def rf_permutation_table(model, X, y, class_names=TARGET_NAMES, feature_names=None,
                         stats: pd.DataFrame | None = None, candidates: int = 100,
                         n_repeats: int = 5, top_n: int = 15, kinds=('word',),
                         max_ngram: int | None = None,
                         random_state: int = 42) -> pd.DataFrame:
    """How much held-out F1 drops when one feature's column is shuffled.

    Fast because only the `candidates` features with the highest impurity
    importance are permuted (a feature the forest never splits on cannot matter).
    Reports the drop in F1 for EACH class and for macro-F1, with the std over
    repeats. Drops near 0 or negative are noise.

    X, y: held-out data (the test matrix), already transformed. For a Pipeline
    you may pass raw text/DataFrame instead.
    """
    clf, names, pre = resolve_model(model, feature_names)
    if not hasattr(clf, 'feature_importances_'):
        raise TypeError("rf_permutation_table needs a fitted tree ensemble.")
    X = _as_matrix(X, pre)
    X = X.toarray() if sp.issparse(X) else np.array(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    kind = _kind(names)
    pool = np.flatnonzero(np.isin(kind, kinds) & _ngram_ok(names, max_ngram))
    pool = pool[np.argsort(clf.feature_importances_[pool])[::-1][:candidates]]

    def f1s(pred):
        return np.array([f1_score(y == c, pred == c, zero_division=0)
                         for c in range(len(class_names))])

    base = f1s(clf.predict(X))
    rows = []
    for j in pool:
        orig = X[:, j].copy()
        drops = []
        for _ in range(n_repeats):
            X[:, j] = rng.permutation(orig)
            drops.append(base - f1s(clf.predict(X)))
        X[:, j] = orig
        drops = np.array(drops)                    
        row = {'feature': names[j], 'kind': kind[j],
               'impurity_importance': clf.feature_importances_[j],
               'drop_macro_f1': drops.mean(axis=1).mean(),
               'drop_macro_f1_std': drops.mean(axis=1).std()}
        for c, cname in enumerate(class_names):
            row[f'drop_f1_{cname}'] = drops[:, c].mean()
        rows.append(row)

    t = pd.DataFrame(rows)
    if stats is not None:
        t = t.merge(stats, on='feature', how='left')
        t['class'] = t['lean']
    else:
        t['class'] = '(all)'
    return _top_per_class(t, 'drop_macro_f1', top_n)

def consensus_words(nb_tbl: pd.DataFrame, rf_tbl: pd.DataFrame) -> dict[str, list[str]]:
    """Words that appear in BOTH the NB and RF top lists, per class. These are
    the safest claims for PS1: two different learners agree."""
    out = {}
    for cls in nb_tbl['class'].unique():
        a = set(nb_tbl.loc[nb_tbl['class'] == cls, 'feature'])
        b = set(rf_tbl.loc[rf_tbl['class'] == cls, 'feature'])
        out[cls] = sorted(a & b)
    return out


def explain_message_nb(model, x_row, feature_names=None, class_names=TARGET_NAMES,
                       top_n: int = 10) -> pd.DataFrame:
    """For ONE message (a single row of the feature matrix), each present
    feature's contribution to the Smishing-vs-Legitimate score:

        contribution = value * (feature_log_prob_[1] - feature_log_prob_[0])

    Positive pushes toward Smishing, negative toward Legitimate. The sum plus
    `.attrs['bias']` equals the model's log-score gap (checked in the tests).
    Binary models only.
    """
    clf, names, _ = resolve_model(model, feature_names)
    if len(class_names) != 2:
        raise ValueError("explain_message_nb is for binary models.")
    x = sp.csr_matrix(x_row)
    if x.shape[0] != 1:
        raise ValueError("Pass exactly one row, e.g. X_test[i].")
    gap = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    contrib = x.data * gap[x.indices]
    out = pd.DataFrame({'feature': names[x.indices], 'value': x.data,
                        'contribution': contrib})
    out['toward'] = np.where(out['contribution'] >= 0, class_names[1], class_names[0])
    out = (out.reindex(out['contribution'].abs().sort_values(ascending=False).index)
              .head(top_n).reset_index(drop=True))
    out.attrs['bias'] = (float(clf.class_log_prior_[1] - clf.class_log_prior_[0])
                         if isinstance(clf, MultinomialNB) else 0.0)
    return out


def prediction_table(model, X, y, texts, class_names=TARGET_NAMES) -> pd.DataFrame:
    """Per-message predictions with class probabilities. Filter `correct == 0`
    and sort by `confidence` to find the confident mistakes."""

    pred = model.predict(X)
    proba = model.predict_proba(X)
    y = np.asarray(y)
    out = pd.DataFrame({'text': np.asarray(texts),
                        'actual': np.array(class_names)[y],
                        'predicted': np.array(class_names)[pred]})
    for c, cname in enumerate(class_names):
        out[f'p_{cname}'] = proba[:, c]
    out['confidence'] = proba.max(axis=1)
    out['correct'] = (pred == y).astype(int)
    return out


def plot_word_table(tbl: pd.DataFrame, value: str = 'score', title: str = '',
                    figsize=(11, 5)):
    """One horizontal bar panel per class from any of the tables above."""

    classes = list(tbl['class'].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=figsize, squeeze=False)
    for ax, cls in zip(axes[0], classes):
        d = tbl[tbl['class'] == cls].iloc[::-1]
        ax.barh(d['feature'], d[value])
        ax.set_title(cls)
        ax.set_xlabel(value)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
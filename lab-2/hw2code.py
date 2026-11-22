import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def find_best_split(feature_vector, target_vector):
    x = np.asarray(feature_vector, dtype=float)
    y = np.asarray(target_vector)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError('Нужны непустые одномерные векторы одинаковой длины')

    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Ожидаются конечные признаки и классы 0/1')

    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    cuts = np.flatnonzero(x[:-1] < x[1:]) + 1

    if not len(cuts):
        return np.array([]), np.array([]), None, None

    thresholds = x[cuts-1]/2 + x[cuts]/2

    thresholds = np.where(thresholds <= x[cuts-1], x[cuts], thresholds)
    positives = np.cumsum(y, dtype=float)
    p_left = positives[cuts-1]/cuts
    p_right = (positives[-1]-positives[cuts-1])/(len(y)-cuts)
    ginis = -(cuts*2*p_left*(1-p_left) + (len(y)-cuts)*2*p_right*(1-p_right))/len(y)
    best = np.argmax(ginis)

    return thresholds, ginis, thresholds[best], ginis[best]


class DecisionTree(ClassifierMixin, BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, X, y, node, depth=0):
        node.update(type='terminal', **{'class': int(np.argmax(np.bincount(y, minlength=2)))})
        
        if (np.all(y == y[0]) or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(y) < (self.min_samples_split or 2)):
            return
        
        best_score, best = -np.inf, None
        
        for feature, kind in enumerate(self.feature_types):
            values = X[:, feature]
            categories = None
            
            if kind == 'categorical':
                categories, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
                positive = np.bincount(inverse, weights=y, minlength=len(categories))
                order = np.argsort(positive/counts, kind='stable')
                ranks = np.empty(len(order), dtype=int)
                ranks[order] = np.arange(len(order))
                values = ranks[inverse]
                categories = categories[order]
            else:
                values = values.astype(float)

            thresholds, scores, _, _ = find_best_split(values, y)

            if not len(thresholds):
                continue

            counts_left = np.searchsorted(np.sort(values), thresholds, side='left')
            valid = (counts_left >= (self.min_samples_leaf or 1)) & (len(y)-counts_left >= (self.min_samples_leaf or 1))

            if not valid.any():
                continue

            candidates = np.flatnonzero(valid)
            idx = candidates[np.argmax(scores[valid])]
            
            if scores[idx] > best_score:
                best_score = scores[idx]
                threshold = thresholds[idx]
                best = feature, threshold, values < threshold, categories

        if best is None:
            return

        feature, threshold, split, categories = best
        node['type'] = 'nonterminal'
        node['feature_split'] = feature

        if categories is None:
            node['threshold'] = threshold
        else:
            node['categories_split'] = set(categories[np.arange(len(categories)) < threshold])

        node['left_child'], node['right_child'] = {}, {}
        self._fit_node(X[split], y[split], node['left_child'], depth+1)
        self._fit_node(X[~split], y[~split], node['right_child'], depth+1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        value = x[node['feature_split']]
        left = value in node['categories_split'] if 'categories_split' in node else float(value) < node['threshold']

        return self._predict_node(x, node['left_child'] if left else node['right_child'])

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError('Нужны двумерный X и непустой y одинаковой длины')

        if len(self.feature_types) != X.shape[1] or any(t not in ('real','categorical') for t in self.feature_types):
            raise ValueError('Для каждого столбца задайте real или categorical')

        for name, lower in [('max_depth',0),('min_samples_split',2),('min_samples_leaf',1)]:
            value = getattr(self,name)
            if value is not None and (not isinstance(value,(int,np.integer)) or value < lower):
                raise ValueError(f'Некорректный {name}')

        for col, kind in enumerate(self.feature_types):
            if kind == 'real' and not np.isfinite(X[:,col].astype(float)).all():
                raise ValueError('Вещественные признаки должны быть конечными')
            if kind == 'categorical' and any(v is None or (isinstance(v,(float,np.floating)) and np.isnan(v)) for v in X[:,col]):
                raise ValueError('Замените пропуски отдельной категорией до обучения')

        self.classes_, encoded_y = np.unique(y, return_inverse=True)

        if len(self.classes_) > 2:
            raise ValueError('Поддерживается только бинарная классификация')

        self.n_features_in_ = X.shape[1]
        self._tree = {}
        self._fit_node(X, encoded_y, self._tree)

        return self

    def predict(self, X):
        check_is_fitted(self, 'classes_')

        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('Число признаков должно совпадать с fit')

        return self.classes_[np.array([self._predict_node(x,self._tree) for x in X],dtype=int)]

    def get_depth(self):
        check_is_fitted(self, 'classes_')

        def depth(node):
            return 0 if node['type']=='terminal' else 1+max(depth(node['left_child']),depth(node['right_child']))

        return depth(self._tree)

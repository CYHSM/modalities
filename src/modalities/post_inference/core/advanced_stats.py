import warnings

import numpy as np
from scipy import spatial, stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


class AdvancedStats:
    @staticmethod
    def permutation_test(group1_acts, group2_acts, n_permutations=1000, test_statistic="mean_diff"):
        group1_array = np.stack(group1_acts)
        group2_array = np.stack(group2_acts)

        combined = np.vstack([group1_array, group2_array])
        n1 = len(group1_array)

        if test_statistic == "mean_diff":
            observed = np.mean(group1_array, axis=0) - np.mean(group2_array, axis=0)
        elif test_statistic == "t":
            observed = stats.ttest_ind(group1_array, group2_array, axis=0)[0]
        else:
            raise ValueError(f"Unknown test statistic: {test_statistic}")

        null_distribution = []

        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_g1 = combined[:n1]
            perm_g2 = combined[n1:]

            if test_statistic == "mean_diff":
                null_stat = np.mean(perm_g1, axis=0) - np.mean(perm_g2, axis=0)
            else:
                null_stat = stats.ttest_ind(perm_g1, perm_g2, axis=0)[0]

            null_distribution.append(null_stat)

        null_distribution = np.array(null_distribution)

        p_values = np.zeros_like(observed)
        for i in range(observed.shape[0]):
            extreme = np.sum(np.abs(null_distribution[:, i]) >= np.abs(observed[i]))
            p_values[i] = (extreme + 1) / (n_permutations + 1)

        return {"observed": observed, "p_values": p_values, "null_distribution": null_distribution}

    @staticmethod
    def multivariate_analysis(group1_acts, group2_acts, method="hotelling"):
        group1_array = np.stack(group1_acts)
        group2_array = np.stack(group2_acts)

        n1, n2 = len(group1_array), len(group2_array)
        p = group1_array.shape[1]

        mean1 = np.mean(group1_array, axis=0)
        mean2 = np.mean(group2_array, axis=0)
        mean_diff = mean1 - mean2

        if method == "hotelling":
            cov1 = np.cov(group1_array.T)
            cov2 = np.cov(group2_array.T)
            pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

            try:
                inv_pooled = np.linalg.inv(pooled_cov)
                t2 = (n1 * n2) / (n1 + n2) * mean_diff @ inv_pooled @ mean_diff

                f_stat = t2 * (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)
                df1, df2 = p, n1 + n2 - p - 1
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                return {"T2": t2, "F": f_stat, "p_value": p_value, "df": (df1, df2), "mean_diff": mean_diff}
            except np.linalg.LinAlgError:
                warnings.warn("Singular covariance matrix, using regularization")
                reg = 1e-6 * np.eye(pooled_cov.shape[0])
                inv_pooled = np.linalg.inv(pooled_cov + reg)
                t2 = (n1 * n2) / (n1 + n2) * mean_diff @ inv_pooled @ mean_diff
                return {"T2": t2, "p_value": np.nan, "mean_diff": mean_diff}

        elif method == "mahalanobis":
            combined = np.vstack([group1_array, group2_array])
            labels = np.array([0] * n1 + [1] * n2)

            distances = []
            for i in range(len(combined)):
                mask = np.arange(len(combined)) != i
                train_data = combined[mask]
                train_labels = labels[mask]

                mean0 = np.mean(train_data[train_labels == 0], axis=0)
                mean1 = np.mean(train_data[train_labels == 1], axis=0)
                cov_pooled = np.cov(train_data.T)

                try:
                    inv_cov = np.linalg.inv(cov_pooled + 1e-6 * np.eye(cov_pooled.shape[0]))
                    if labels[i] == 0:
                        dist = spatial.distance.mahalanobis(combined[i], mean0, inv_cov)
                    else:
                        dist = spatial.distance.mahalanobis(combined[i], mean1, inv_cov)
                    distances.append(dist)
                except:
                    distances.append(np.nan)

            _, p_value = stats.mannwhitneyu(
                [d for d, l in zip(distances, labels) if l == 0 and not np.isnan(d)],
                [d for d, l in zip(distances, labels) if l == 1 and not np.isnan(d)],
            )

            return {"distances": distances, "p_value": p_value, "mean_diff": mean_diff}

    @staticmethod
    def representational_similarity_analysis(*, activations_dict, distance_metric="correlation"):
        conditions = list(activations_dict.keys())
        n_conditions = len(conditions)
        rdm = np.zeros((n_conditions, n_conditions))

        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                acts1 = np.mean(activations_dict[cond1], axis=0).flatten()
                acts2 = np.mean(activations_dict[cond2], axis=0).flatten()

                if distance_metric == "correlation":
                    rdm[i, j] = 1 - np.corrcoef(acts1, acts2)[0, 1]
                elif distance_metric == "euclidean":
                    rdm[i, j] = np.linalg.norm(acts1 - acts2)
                elif distance_metric == "cosine":
                    rdm[i, j] = spatial.distance.cosine(acts1, acts2)

        return {"rdm": rdm, "conditions": conditions, "metric": distance_metric}

    @staticmethod
    def decoding_analysis(*, group1_acts, group2_acts, method="lda", cv_folds=5):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import SVC

        X = np.vstack([group1_acts, group2_acts])
        y = np.array([0] * len(group1_acts) + [1] * len(group2_acts))

        if method == "lda":
            clf = LinearDiscriminantAnalysis()
        elif method == "svm":
            clf = SVC(kernel="linear", probability=True)
        elif method == "rf":
            clf = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown method: {method}")

        scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc")

        clf.fit(X, y)
        y_pred_proba = clf.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, y_pred_proba)

        feature_importance = None
        if method == "lda":
            feature_importance = np.abs(clf.coef_[0])
        elif method == "svm":
            if hasattr(clf, "coef_"):
                feature_importance = np.abs(clf.coef_[0])
        elif method == "rf":
            feature_importance = clf.feature_importances_

        return {
            "cv_scores": scores,
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "train_auc": train_auc,
            "feature_importance": feature_importance,
            "classifier": clf,
        }

    @staticmethod
    def pca_analysis(*, activations_list, n_components=10):
        X = np.vstack(activations_list)

        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_transformed = pca.fit_transform(X)

        return {
            "components": pca.components_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "transformed": X_transformed,
            "pca": pca,
        }

    @staticmethod
    def temporal_analysis(*, activation_sequence, window_size=5):
        n_steps = len(activation_sequence)

        mean_trajectory = []
        std_trajectory = []
        autocorrelation = []

        for acts in activation_sequence:
            flat = acts.flatten()
            mean_trajectory.append(np.mean(flat))
            std_trajectory.append(np.std(flat))

        mean_trajectory = np.array(mean_trajectory)

        for lag in range(1, min(window_size, n_steps)):
            if lag < n_steps:
                corr = np.corrcoef(mean_trajectory[:-lag], mean_trajectory[lag:])[0, 1]
                autocorrelation.append((lag, corr))

        differences = []
        for i in range(1, n_steps):
            prev = activation_sequence[i - 1].flatten()
            curr = activation_sequence[i].flatten()
            diff = np.mean(np.abs(curr - prev))
            differences.append(diff)

        return {
            "mean_trajectory": mean_trajectory,
            "std_trajectory": std_trajectory,
            "autocorrelation": autocorrelation,
            "step_differences": differences,
            "mean_change_rate": np.mean(differences) if differences else 0,
        }

    @staticmethod
    def cluster_analysis(*, activations, n_clusters=5):
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.metrics import silhouette_score

        X = np.vstack(activations)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_silhouette = silhouette_score(X, kmeans_labels)

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

        dbscan_silhouette = None
        if n_dbscan_clusters > 1:
            valid_mask = dbscan_labels != -1
            if np.sum(valid_mask) > 1:
                dbscan_silhouette = silhouette_score(X[valid_mask], dbscan_labels[valid_mask])

        return {
            "kmeans": {
                "labels": kmeans_labels,
                "centers": kmeans.cluster_centers_,
                "silhouette": kmeans_silhouette,
                "n_clusters": n_clusters,
            },
            "dbscan": {
                "labels": dbscan_labels,
                "n_clusters": n_dbscan_clusters,
                "n_noise": np.sum(dbscan_labels == -1),
                "silhouette": dbscan_silhouette,
            },
        }

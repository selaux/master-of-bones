import numpy as np
from itertools import combinations
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.tree import tree

def find_measurable_differences(bones, class1, class2, landmark_extractor, tree_criterion, tree_max_depth, tree_min_samples_leaf, n_folds, progress_callback=None):
    for bi, bone in enumerate(bones):
        marker_combinations = combinations(range(7), 2)
        bone['landmarks'] = landmark_extractor(bone['points'])
        bone['landmark_distances'] = {}
        for i, j in marker_combinations:
            bone['landmark_distances'][(i, j)] = np.linalg.norm(bone['landmarks'][i, :] - bone['landmarks'][j, :])
        progress_callback(bi, len(bones)-1)

    cls1, cls2 = class1, class2

    first_outline = bones[0]
    combs = list(combinations(first_outline['landmark_distances'].keys(), 2))
    distance_combinations = [(dist1, dist2) for dist1, dist2 in combs]

    classes = np.array(map(lambda o: 1 if o['class'] == cls1 else -1, bones))
    features = []
    for bone in bones:
        distances = bone['landmark_distances']
        feature = np.array([distances[dist1] / distances[dist2] for dist1, dist2 in distance_combinations])
        features.append(feature)
    features = np.array(features)

    clf = tree.DecisionTreeClassifier(
        criterion=tree_criterion,
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf
    )

    skf = cross_validation.StratifiedKFold(classes, n_folds=n_folds)
    scores = []
    for train_indices, test_indices in skf:
        print(train_indices, test_indices)
        features_train, classes_train = features[train_indices], classes[train_indices]
        features_test, classes_test = features[test_indices], classes[test_indices]

        fitted = clf.fit(features_train, classes_train)
        #scores.append(f1_score(classes_test, clf.predict(features_test)))
        scores.append(fitted.score(features_test, classes_test))
    scores = np.array(scores)
    #scores = map(lambda d: clf.fit(features[d[0], :], classes[d[0]]).score(features[d[1], :], classes[d[1]]), skf)
    #scores = cross_validation.cross_val_score(clf, features, classes, cv=10)
    mean_cv_score = scores.mean()
    cv_confidence_interval = scores.std() * 2

    clf.fit(features, classes)

    return {
        'landmark_combinations': list(combs),
        'decision_tree': clf,
        'mean_cv_score': mean_cv_score,
        'cv_confidence_interval': cv_confidence_interval
    }

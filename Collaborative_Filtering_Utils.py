import pandas as pd
import numpy as np
from collections import defaultdict

import Utils as utils

from sklearn.model_selection import train_test_split as train_test_split_sklearn
from surprise import Reader, Dataset
from surprise import SVD, SVDpp, SlopeOne, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import KFold, cross_validate, GridSearchCV, train_test_split


def get_top_n(predictions, articles, article_indices, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(information, est), ...] of size n. Information consists of title and domain name
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        article = articles.loc[iid]
        information = article['title'] + ' on ' + article['domain_name']
        top_n[uid].append((information, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        k: The number of relevant results among the top k documents 
        threshold: Estimation threshold
        
    Returns:
        precison: Proportion of recommended items that are relevant
        recall: Proportion of relevant items that are recommended
    """

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

def prepare_train_test_splits_cross_validate(df, is_normalized):
    """Create train and test data for cross validation

    Args:
        df: Users data
        is_normalized: Whether virality column is normalized
    Returns:
        train_data_object: Dataset object from Surprise
        test_data_object: Dataset object from Surprise
        user_interactions_data_object: Dataset object from Surprise. Complete users data for kFold

    """
    if is_normalized:
        scale=(0.0, 1.0)
    else:
        scale=(0.0, 10000000.0)
        
        
    reader = Reader(rating_scale=scale)
    
    # Used in kFold
    user_interactions_data_object = Dataset.load_from_df(df[['personId', 'contentId', 'virality']], reader)
    
    # Create train and test splits using sklearn. These splits are to be used in cross validation
    train_df, test_df = train_test_split_sklearn(df,
                                   stratify=df['personId'], 
                                   test_size=0.20,
                                   random_state=20)
    
    
    train_data_object = Dataset.load_from_df(train_df[['personId', 'contentId', 'virality']], reader)
    test_data_object = Dataset.load_from_df(test_df[['personId', 'contentId', 'virality']], reader)
    
    
    return train_data_object, test_data_object, user_interactions_data_object


def prepare_train_test_sets(train_data_object, test_data_object):
    """Create train and test data for fit() and test() methods

    Args:
        train_data_object: Dataset object from Surprise
        test_data_object: Dataset object from Surprise
    Returns:
        built_full_train: Train set
        anti_test_set_from_train: Unknown articles from train set
        built_full_test_set: Test set
        train_to_test_set: Train set wrapped in build_testset(). To test train set's performance

    """
    # Build full trainset - use in fit() - Whole sets
    built_full_train = train_data_object.build_full_trainset()

    #unknown test set. user in test()
    built_full_test = test_data_object.build_full_trainset()

    # Comes from train set
    anti_test_set_from_train = built_full_train.build_anti_testset()
    
    # Comes from test split
    built_full_test_set = built_full_test.build_testset()
    
    # Create test sets from train set, use in test()
    train_to_test_set = built_full_train.build_testset()
    
    return built_full_train, anti_test_set_from_train, built_full_test_set, train_to_test_set


def plot_model_rmse(xs, ys, title, x_label, y_label):
    """ Helper function to plot RMSE for each 

    Args:
        train_data_object: Dataset object from Surprise
        test_data_object: Dataset object from Surprise
    Returns:
        built_full_train: Train set
        anti_test_set_from_train: Unknown articles from train set
        built_full_test_set: Test set
        train_to_test_set: Train set wrapped in build_testset(). To test train set's performance

    """
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(xs, ys, marker = 'o')
    
    for x,y in zip(xs,ys):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title(title, fontsize = 12)
    plt.xlabel(x_label, fontsize = 10)
    plt.ylabel(y_label, fontsize = 10)
    plt.draw()


def algorithm_search(trainset):
    """ Searches for a CF algorihm from Surprise

    Args:
        trainset: Dataset object from Surprise
    Returns:
        result_df: CV result of all algorithms

    """
    benchmark = []

    # Iterate over all algorithms
    for algorithm in [SVD(), SVDpp(), SlopeOne(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # Use all performance measures
        for measure in ['RMSE', 'MSE', 'MAE']:
            # Perform cross validation
            results = cross_validate(algorithm, trainset, measures=[measure], cv=3, verbose=False)

            # Get results & append algorithm name
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)

    result_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    
    return result_df

def precision_recall_k_fold(fold, algorithm, complete_data, at_k, thr):
    """ Runs a k-Fold CV. Computes Precision@k and Recall@k for each CV run

    Args:
        fold: k-Fold number
        algoritm: Surprise algorithm instance()
        complete_data: Dataset object from Surprise
        at_k: The number of relevant results among the top k documents 
        thr: Estimation threshold
    Returns:
        None

    """
    precision = []
    recall = []
    kf = KFold(n_splits=fold)
    for trainset, testset in kf.split(complete_data):
        algorithm.fit(trainset)
        predictions = algorithm.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=at_k, threshold=thr)

        # Precision and recall can then be averaged over all users
        p = sum(prec for prec in precisions.values()) / len(precisions)
        r = sum(rec for rec in recalls.values()) / len(recalls)
        print('Precision at', at_k, p)
        print('Recall at', at_k, r)
        precision.append(p)
        recall.append(r)

    return sum(precision)/at_k, sum(recall)/at_k


def print_top_n_predictions_per_user(predictions, articles_df, article_indices, n):
    """ Prints top n predictions for each user

    Args:
        predictions: Predictions
        articles_df: Articles data
        article_indices: Article indices
        n: top n predictions
    Returns:
        None

    """
    top_n = get_top_n(predictions, articles_df, article_indices, n=n)
    for uid, user_ratings in top_n.items():
        print('User ', uid)

        for recommendations in user_ratings:
            print(recommendations[0])

        print('  ')

def grid_search(algorithm, parameters, train_data):
    """ Prints top n predictions for each user

    Args:
        algorithm: Algorithm
        parameters: parameter set
        train_data: train data
    Returns:
        None

    """
    gs = GridSearchCV(algorithm, parameters, measures=['rmse', 'mae'], cv=3)
    print(type(gs))
    gs.fit(train_data)

    # best RMSE score
    print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    results_df = pd.DataFrame.from_dict(gs.cv_results)
    
    return results_df
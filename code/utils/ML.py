# Note to self: add ensemble learner

import numpy as np
import multiprocessing as mp
num_cores = round(mp.cpu_count()/1.5)

##########################################################
# Models with parameters for different learners
##########################################################

def ModelParams(learner, cv = 10, nj = 1):

    # Logistic regression
    if learner == 'logit':
        label = 'Logistic Regression'
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    
    # Logistic-Lasso with CV
    elif learner == 'lasso':
        label = 'Logistic Lasso'
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(max_iter=1000, penalty='l1', 
                                     solver='saga', cv = cv, n_jobs=nj)
        
    # Regression Forest
    elif learner == 'rf':
        label = 'Random Forest Classifier'
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        param_grid = {'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40]}
        model = GridSearchCV(RandomForestClassifier(n_estimators=100, n_jobs=nj),
                          param_grid, cv=cv, n_jobs=nj)
    
    # Neural Network
    elif learner == 'nn':
        label = 'Neural Network'
        from sklearn.model_selection import GridSearchCV
        from sklearn.neural_network import MLPClassifier
        parameter_space = {
            'hidden_layer_sizes': [(5,), (10,), (20,), (5, 5), (10, 10), 
                                   (20, 20), (5, 5, 5), (10, 10, 10)]
        }
        model = GridSearchCV(MLPClassifier(max_iter=1000, solver='adam'), 
                             parameter_space, cv=cv, n_jobs=nj)
    
    # Raise error if invalid learner
    elif learner not in ['logit', 'lasso', 'rf', 'nn']:
        raise ValueError('Invalid learner. Choose from logit, lasso, rf, nn')
    
    return model, label

##########################################################
# Picks the best model from a selection of learners
##########################################################

def BestModel(X, Y, test_size = 0.3, cv = 10, print_opt = 'verbose'):

    # Initialize
    np.random.seed(1118)
    learners = ['logit', 'lasso', 'rf', 'nn']
    auc_list, labels, models = [], [], []
    pars = 'No additional parameters'

    # Split data into training and testing
    from sklearn.model_selection import train_test_split as tts
    X_trn, X_tst, Y_trn, Y_tst = tts(X, Y, test_size=test_size)

    # Fit models
    for learner in learners:

        # Fit model, calculate AUC, & append lists
        model, label = ModelParams(learner, cv)
        model.fit(X_trn, Y_trn)
        auc = float(model.score(X_tst, Y_tst))
        labels.append(label)
        models.append(model)
        auc_list.append(auc)

        # Print output
        if learner == 'lasso':
            pars = 'Optimal alpha: {:.2f}'.format(model.C_[0])
        elif learner == 'rf' or learner == 'nn':
            pars = 'Optimal parameters: {}'.format(model.best_params_)
        if print_opt == 'verbose':
            print('\n' , label, 'AUC: {:.3f}'.format(auc), f'({pars})')

    # Find the best model
    best_model_idx = np.argmax(auc_list)
    best_model = models[best_model_idx]
    print('\n', 'Best model:', labels[best_model_idx], 
          'with AUC {:.3f}'.format(auc_list[best_model_idx]))
    
    return best_model

##########################################################
import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from hyperopt import tpe, fmin, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import mord
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

sns.set(style='darkgrid', font_scale=2)
warnings.filterwarnings('ignore')
random.seed(829)
np.random.seed(829)


def query_data(path='main.dta'):
    """
    Query the data and split it into three parts by its features -- 1) focal only; 2) supply chain only; 3) focal and
    supply chain.
    :param path: file path to read the data from
    :return: three datasets which can be modeled on later
    """
    df = pd.read_stata(path)
    df.dropna(how='all', axis=1, inplace=True)
    # df = df.sample(frac=1).reset_index(drop=True)

    # Focal data
    df_f = df.filter(regex='^f_|_centr')
    df_f['sector'] = df['sector']
    df_f.drop(columns=['f_cusip'], inplace=True)

    # Supply chain data
    df_sc = df.filter(regex='^s_|^sw_|^c_|^cw_|_centr')
    df_sc['sector'] = df['sector']
    df_sc.drop(columns=['c_cusip'], inplace=True)

    # Focal and supply chain data
    df_fsc = df.filter(regex='^f_|^s_|^sw_|^c_|^cw_|_centr')
    df_fsc['sector'] = df['sector']
    df_fsc.drop(columns=['c_cusip', 'f_cusip'], inplace=True)

    y = df['rgroup_code']

    return df_f, df_sc, df_fsc, y


def feature_selection(x, y, model, num=6, method='shap'):
    if method == 'shap':
        try:
            importance = np.abs(shap.TreeExplainer(model).shap_values(x)).mean(axis=0)
        except Exception:
            importance = np.abs(shap.LinearExplainer(model, x).shap_values(x)).mean(axis=0)
        importance = pd.Series(importance.mean(axis=0), index=x.columns)
        importance.sort_values(ascending=False, inplace=True)
        # pd.set_option('display.max_rows', None)
        # print(importance)

    if method == 'permutation':
        importance = permutation_importance(model, x, y, random_state=0)
        importance = pd.Series(importance.importances_mean, index=x.columns)
        importance.sort_values(ascending=False, inplace=True)

    if method == 'lasso':
        importance = Lasso(normalize=True).fit(x, y)
        importance = pd.Series(importance.coef_, index=x.columns)
        importance.sort_values(ascending=False, inplace=True)

    if method == 'pca':
        x_scaled = StandardScaler().fit_transform(x)
        importance = PCA(n_components=1).fit(x_scaled)
        importance = pd.Series(importance.components_[0], index=x.columns)
        importance.sort_values(ascending=False, inplace=True)

    def drop_duplicate(cols, num):
        var_set = set()
        var_list = []
        for col in cols:
            name = col.replace('sw_', 's_')
            name = name.replace('cw_', 'c_')
            if name not in var_set:
                var_set.add(name)
                var_list.append(col)
            if len(var_list) == num:
                return var_list
        return var_list

    return drop_duplicate(importance.index, num)


def hp_tuning(x, y, space_model='lgb'):
    if space_model == 'lgb' or space_model == 'xgb':
        space = {'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),
                 'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 1)),
                 'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                 'num_leaves': 10}

    elif space_model == 'svm':
        space = {'C': hp.qloguniform('C', -2, 0, 0.05)}

    elif space_model == 'rf':
        space = {'max_features': scope.int(hp.quniform('max_features', 1, len(x.columns) - 1, 1))}

    elif space_model == 'knn':
        space = {'n_neighbors': scope.int(hp.quniform('n_neighbors', 1, 51, 1))}

    elif space_model == 'ada':
        space = {'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10))}

    else:
        print("Tuning not supported.")

    def para_searching(params):
        model = LGBMClassifier(**params, n_jobs=-1, random_state=508)
        acc = cross_val_score(model, x, y, cv=10, scoring='accuracy').mean()
        return {'loss': -acc, 'status': STATUS_OK}

    rstate = np.random.RandomState(509)
    trials = Trials()
    best = fmin(fn=para_searching, space=space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=rstate)
    print("The best parameter combo is : {}.".format(space_eval(space, best)))
    return space_eval(space, best)


def within_one_notch_ar(y_true, y_pred):
    diff = abs(y_true-y_pred)
    within = np.sum(diff <= 1)
    num = len(y_true)
    return within/num


def model_eval(model, train_x, train_y, test_x, test_y, x, y, display=True):
    pred_y = model.predict(test_x)
    train_pred = model.predict(train_x)
    pred_proba = model.predict_proba(test_x)
    train_acc = model.score(train_x, train_y)
    train_war = within_one_notch_ar(train_y, train_pred)
    acc = accuracy_score(y_true=test_y, y_pred=pred_y)
    war = within_one_notch_ar(y_true=test_y, y_pred=pred_y)
    cm = confusion_matrix(y_true=test_y, y_pred=pred_y)
    cm_train = confusion_matrix(y_true=train_y, y_pred=train_pred)
    f_1 = f1_score(y_true=test_y, y_pred=pred_y, average='weighted')
    try:
        auc = roc_auc_score(y_true=test_y, y_score=pred_proba, average='weighted', multi_class='ovr')
    except ValueError:
        auc = 1

    cv_acc = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=508)
    for train_idx, test_idx in skf.split(x, y):
        tr_x, tr_y = x[x.index.isin(train_idx)], y[y.index.isin(train_idx)]
        te_x, te_y = x[x.index.isin(test_idx)], y[y.index.isin(test_idx)]
        tmp = model.fit(tr_x, tr_y)
        # pred = model.predict(te_x)
        cv_acc.append(tmp.score(te_x, te_y))
    cv_acc = np.mean(cv_acc)

    if display:
        # print("The train accuracy is : {0:.4f}.".format(train_acc))
        # print("The test accuracy is : {0:.4f}.".format(acc))
        # print("The within one notch AR is : {0:.4f}.".format(war))
        # print("The test F1 Score is : {0:.4f}.".format(f_1))
        # print("The test AUC is : {0:.4f}.".format(auc))
        # print("The test confusion matrix is\n", cm)
        # print("The train confusion matrix is\n", cm_train)
        # print("The 5-fold CV accuracy is {0:.4f}.".format(cv_acc))
        print(str(acc)+","+str(war))

    return train_acc, acc, war, f_1, auc, train_war, cv_acc


def model_main(x, y, random_state=1, fs=True, test_size=0.2, rstate=418, num=6, tuning=False, display=False,
              fs_method='shap', model='lgb'):
    """
    Fit a LightGBM model with the data.
    :param x: attributes
    :param y: labels
    :param random_state: for replication
    :param fs: (boolean type) whether to perform feature selection
    :param test_size: test size for train test split
    :param rstate: for replication in train test split
    :param num: number of attributes kept in the feature selection
    :param tuning: whether to run parameter tuning
    :param display: whether to display the model evaluation metrics
    :return: model evaluation metrics and the used features
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rstate)

    # 1. Train the model.
    if model == 'lgb':
        clf = LGBMClassifier().fit(x_train, y_train)

    if model == 'mda':
        clf = LinearDiscriminantAnalysis().fit(x_train, y_train)

    if model == 'ologit':
        clf = mord.linear_model.LogisticRegression().fit(x_train, y_train)

    if model == 'svm':
        clf = SVC(kernel='linear', probability=True).fit(x_train, y_train)

    if model == 'rf':
        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1).fit(x_train, y_train)

    if model == 'knn':
        clf = KNeighborsClassifier().fit(x_train, y_train)

    if model == 'xgb':
        clf = XGBClassifier(random_state=random_state, n_jobs=-1).fit(x_train, y_train)

    if model == 'ada':
        clf = AdaBoostClassifier(random_state=random_state).fit(x_train, y_train)

    if model == 'mlp':
        clf = MLPClassifier(random_state=random_state).fit(x_train, y_train)

    features = x.columns
    if fs:
        features = feature_selection(x_train, y_train, clf, num, method=fs_method)
        # print(features)
        x_train, x_test, x = x_train.filter(features), x_test.filter(features), x.filter(features)
        clf = clf.fit(x_train, y_train)

    # 2. Tune the parameters.
    if tuning:
        parameters = hp_tuning(x_train, y_train, space_model=model)
        if model == 'lgb':
            clf = LGBMClassifier(random_state=random_state, n_jobs=-1, **parameters).fit(x_train, y_train)

        if model == 'mda':
            clf = LinearDiscriminantAnalysis(**parameters).fit(x_train, y_train)

        if model == 'ologit':
            clf = mord.linear_model.LogisticRegression(**parameters).fit(x_train, y_train)

        if model == 'svm':
            clf = SVC(kernel='linear', probability=True, **parameters).fit(x_train, y_train)

        if model == 'rf':
            clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **parameters).fit(x_train, y_train)

        if model == 'knn':
            clf = KNeighborsClassifier(**parameters).fit(x_train, y_train)

        if model == 'xgb':
            clf = XGBClassifier(random_state=random_state, n_jobs=-1, **parameters).fit(x_train, y_train)

        if model == 'ada':
            clf = AdaBoostClassifier(random_state=random_state, **parameters).fit(x_train, y_train)

        if model == 'mlp':
            clf = MLPClassifier(random_state=random_state, **parameters).fit(x_train, y_train)

    # 3. Evaluate the model.
    train_acc, acc, war, f_1, auc, train_war, cv_acc = model_eval(clf, x_train, y_train, x_test, y_test, x, y, display=display)
    return train_acc, acc, features, war, f_1, auc, train_war, cv_acc


def model_by_sector(x, y, random_state=1, fs=True, test_size=0.2, rstate=418, num=6, tuning=False, display=True):
    sector_dict = {}
    for sec in x['sector'].unique():
        print(sec)
        tmp_x = x[x['sector']==sec].drop(columns=['sector'])
        tmp_x.dropna(axis=1, inplace=True)
        tmp_y = y[y.index.isin(tmp_x.index)]
        tmp_x, tmp_y = tmp_x.reset_index(drop=True), tmp_y.reset_index(drop=True)
        shape = len(tmp_x)
        train_acc, acc, features, war, f_1, auc, train_war, cv_acc = model_main(tmp_x, tmp_y, random_state, fs, test_size, rstate, num, 
                                                             tuning, display)
        sector_dict.update({sec: (shape, train_acc, acc, features, war, f_1, auc, train_war, cv_acc)})

    train_acc_all = np.average([sector_dict[k][1] for k in sector_dict],
                               weights=[sector_dict[k][0] for k in sector_dict])
    acc_all = np.average([sector_dict[k][2] for k in sector_dict], weights=[sector_dict[k][0] for k in sector_dict])
    features_all = {k: sector_dict[k][3] for k in sector_dict}
    war_all = np.average([sector_dict[k][4] for k in sector_dict], weights=[sector_dict[k][0] for k in sector_dict])
    f1_all = np.average([sector_dict[k][5] for k in sector_dict], weights=[sector_dict[k][0] for k in sector_dict])
    auc_all = np.average([sector_dict[k][6] for k in sector_dict], weights=[sector_dict[k][0] for k in sector_dict])
    train_war_all = np.average([sector_dict[k][7] for k in sector_dict], weights=[sector_dict[k][0] for k in sector_dict])
    cv_acc_all = np.average([sector_dict[k][8] for k in sector_dict], weights=[sector_dict[k][0] for k in sector_dict])

    return train_acc_all, acc_all, features_all, war_all, f1_all, auc_all, train_war_all, cv_acc_all


if __name__ == '__main__':
    x_f, x_sc, x_fsc, y = query_data()
    tuning = False

    print("Focal only:")
    train_acc_f, acc_f, focal_feat, war_f, f1_f, auc_f, train_war_f, cv_acc_f = model_main(x_f.drop(columns=['sector']), y, fs=True, num=6, tuning=tuning)
    train_acc_all_f, acc_all_f, features_all_f, war_all_f, f1_all_f, auc_all_f, train_war_all_f, cv_acc_all_f = model_by_sector(x_f, y, tuning=False)
    # print(train_acc_all_f, acc_all_f)

    print("\nSupply chain only:")
    train_acc_sc, acc_sc, sc_feat, war_sc, f1_sc, auc_sc, train_war_sc, cv_acc_sc = model_main(x_sc.drop(columns=['sector']), y, num=12, tuning=tuning)
    train_acc_all_sc, acc_all_sc, features_all_sc, war_all_sc, f1_all_sc, auc_all_sc, train_war_all_sc, cv_acc_all_sc = model_by_sector(x_sc, y, num=12, tuning=False)
    # print(train_acc_all_sc, acc_all_sc)

    print("\nFocal and supply chain:")
    fsc_feat = focal_feat+sc_feat
    train_acc_fsc, acc_fsc, _, war_fsc, f1_fsc, auc_fsc, train_war_fsc, cv_acc_fsc = model_main(x_fsc.filter(fsc_feat), y, fs=False, tuning=tuning)

    fsc_feat_by_sector = {k: features_all_f[k]+features_all_sc[k] for k in features_all_f}
    x_fsc_by_sector = pd.DataFrame()
    for sec in x_fsc['sector'].unique():
        tmp = x_fsc[x_fsc['sector']==sec].filter(fsc_feat_by_sector[sec])
        tmp['sector'] = sec
        x_fsc_by_sector = pd.concat([x_fsc_by_sector, tmp])
    train_acc_all_fsc, acc_all_fsc, _, war_all_fsc, f1_all_fsc, auc_all_fsc, train_war_all_fsc, cv_acc_all_fsc = model_by_sector(x_fsc_by_sector, y, fs=False, tuning=False)
    # print(train_acc_all_fsc, acc_all_fsc)

    print("\nSupplier only:")
    x_s = x_sc.filter(sc_feat)
    x_s = x_s.filter(regex='^s_|^sw_')
    x_s['sector'] = x_sc['sector']
    train_acc_s, acc_s, s_feat, war_s, f1_s, auc_s, train_war_s, cv_acc_s = model_main(x_s.drop(columns=['sector']), y, fs=False, tuning=tuning)
    train_acc_all_s, acc_all_s, features_all_s, war_all_s, f1_all_s, auc_all_s, train_war_all_s, cv_acc_all_s = model_by_sector(x_s, y, tuning=False)
    # print(train_acc_all_s, acc_all_s)

    print("\nCustomer only:")
    x_c = x_sc.filter(sc_feat)
    x_c = x_c.filter(regex='^c_|^cw_')
    x_c['sector'] = x_sc['sector']
    train_acc_c, acc_c, c_feat, war_c, f1_c, auc_c, train_war_c, cv_acc_c = model_main(x_c.drop(columns=['sector']), y, fs=False, tuning=tuning)
    train_acc_all_c, acc_all_c, features_all_c, war_all_c, f1_all_c, auc_all_c, train_war_all_c, cv_acc_all_c = model_by_sector(x_c, y, tuning=False)
    # print(train_acc_all_c, acc_all_c)

    results = pd.DataFrame(index=['f', 'sc', 'fsc', 's', 'c', 'f_by', 'sc_by', 'fsc_by', 's_by', 'c_by'], 
                           columns=['train AR', 'AR', 'WAR', 'F1', 'AUC', 'train WAR', 'CV AR'])

    results.loc['f', 'train AR'] = train_acc_f
    results.loc['sc', 'train AR'] = train_acc_sc
    results.loc['fsc', 'train AR'] = train_acc_fsc
    results.loc['s', 'train AR'] = train_acc_s
    results.loc['c', 'train AR'] = train_acc_c

    results.loc['f_by', 'train AR'] = train_acc_all_f
    results.loc['sc_by', 'train AR'] = train_acc_all_sc
    results.loc['fsc_by', 'train AR'] = train_acc_all_fsc
    results.loc['s_by', 'train AR'] = train_acc_all_s
    results.loc['c_by', 'train AR'] = train_acc_all_c
    
    results.loc['f', 'AR'] = acc_f
    results.loc['sc', 'AR'] = acc_sc
    results.loc['fsc', 'AR'] = acc_fsc
    results.loc['s', 'AR'] = acc_s
    results.loc['c', 'AR'] = acc_c

    results.loc['f_by', 'AR'] = acc_all_f
    results.loc['sc_by', 'AR'] = acc_all_sc
    results.loc['fsc_by', 'AR'] = acc_all_fsc
    results.loc['s_by', 'AR'] = acc_all_s
    results.loc['c_by', 'AR'] = acc_all_c

    results.loc['f', 'WAR'] = war_f
    results.loc['sc', 'WAR'] = war_sc
    results.loc['fsc', 'WAR'] = war_fsc
    results.loc['s', 'WAR'] = war_s
    results.loc['c', 'WAR'] = war_c

    results.loc['f_by', 'WAR'] = war_all_f
    results.loc['sc_by', 'WAR'] = war_all_sc
    results.loc['fsc_by', 'WAR'] = war_all_fsc
    results.loc['s_by', 'WAR'] = war_all_s
    results.loc['c_by', 'WAR'] = war_all_c

    results.loc['f', 'F1'] = f1_f
    results.loc['sc', 'F1'] = f1_sc
    results.loc['fsc', 'F1'] = f1_fsc
    results.loc['s', 'F1'] = f1_s
    results.loc['c', 'F1'] = f1_c

    results.loc['f_by', 'F1'] = f1_all_f
    results.loc['sc_by', 'F1'] = f1_all_sc
    results.loc['fsc_by', 'F1'] = f1_all_fsc
    results.loc['s_by', 'F1'] = f1_all_s
    results.loc['c_by', 'F1'] = f1_all_c

    results.loc['f', 'AUC'] = auc_f
    results.loc['sc', 'AUC'] = auc_sc
    results.loc['fsc', 'AUC'] = auc_fsc
    results.loc['s', 'AUC'] = auc_s
    results.loc['c', 'AUC'] = auc_c

    results.loc['f_by', 'AUC'] = auc_all_f
    results.loc['sc_by', 'AUC'] = auc_all_sc
    results.loc['fsc_by', 'AUC'] = auc_all_fsc
    results.loc['s_by', 'AUC'] = auc_all_s
    results.loc['c_by', 'AUC'] = auc_all_c

    results.loc['f', 'train WAR'] = train_war_f
    results.loc['sc', 'train WAR'] = train_war_sc
    results.loc['fsc', 'train WAR'] = train_war_fsc
    results.loc['s', 'train WAR'] = train_war_s
    results.loc['c', 'train WAR'] = train_war_c

    results.loc['f_by', 'train WAR'] = train_war_all_f
    results.loc['sc_by', 'train WAR'] = train_war_all_sc
    results.loc['fsc_by', 'train WAR'] = train_war_all_fsc
    results.loc['s_by', 'train WAR'] = train_war_all_s
    results.loc['c_by', 'train WAR'] = train_war_all_c

    results.loc['f', 'CV AR'] = cv_acc_f
    results.loc['sc', 'CV AR'] = cv_acc_sc
    results.loc['fsc', 'CV AR'] = cv_acc_fsc
    results.loc['s', 'CV AR'] = cv_acc_s
    results.loc['c', 'CV AR'] = cv_acc_c

    results.loc['f_by', 'CV AR'] = cv_acc_all_f
    results.loc['sc_by', 'CV AR'] = cv_acc_all_sc
    results.loc['fsc_by', 'CV AR'] = cv_acc_all_fsc
    results.loc['s_by', 'CV AR'] = cv_acc_all_s
    results.loc['c_by', 'CV AR'] = cv_acc_all_c

    results.to_excel('tmp/results.xlsx')

#!/usr/bin/python
import numpy as np
from IPython.display import display
import eli5
import matplotlib.pyplot as plt
import scipy.sparse
import pickle
import psycopg2
import pandas
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt

plt.style.use('ggplot')

np.set_printoptions(threshold=np.inf)

try:
    # conn = psycopg2.connect(dbname='catarse_db', user='catarse', host='localhost', password='example', port='5445')
    conn = psycopg2.connect(dbname='catarse_production', user='catarse', host='db.catarse.me')
except:
    print(  "I am unable to connect to the database" )

cur = conn.cursor()

f_names = [
        'category_count',
        'mode_count',
        'same_state',
        'recommended',
        'online_days',
        'goal',
        'budget',
        'description',
        'pledged',
        'contributions',
        'owner_projects',
        'reward_count'
    ]

def get_predictions(user_id):
    cur.execute("""
     select
    (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = p.category_id) AS category_count,
     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.mode = p.mode) AS mode_count,
 (coalesce(
            (SELECT state_id
             FROM cities
             WHERE cities.id = p.city_id
             LIMIT 1), 0) = coalesce(
            (SELECT state_id
             FROM addresses a
             WHERE a.id = u.address_id
               AND state_id IS NOT NULL
             LIMIT 1), 0))::integer AS same_state,
    coalesce(p.recommended, false)::integer as recommended,
    coalesce(p.online_days, 60)::integer AS online_days,
    coalesce(p.goal::float, 0) AS goal,
    coalesce(char_length(p.budget), 0) AS budget_length,
    coalesce(char_length(p.about_html), 0) AS about_length,
    COALESCE((SELECT sum(pa.value) from contributions c join payments pa on pa.contribution_id = c.id where pa.state IN ('paid', 'pending') and c.project_id = p.id and pa.created_at <= p.created_at + '3 days'::interval)::float, 0) as pledged,
    (SELECT count(*) from contributions c where c.project_id = p.id and c.created_at <= p.created_at + '3 days'::interval) as total_contributions,
    (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft')::integer as project_count,
    (SELECT count(*) from rewards r where r.project_id = p.id)::integer as reward_count,
    p.id
    from projects p,
    users u
    WHERE u.id = """ + str( user_id ) + """
    and p.state = 'online'
    AND NOT EXISTS (select true from contributions c2 where c2.project_id = p.id and c2.user_id = u.id)
    --and p.category_id = 14
   and p.id IN( 72320)
    """)

    rows = np.array(cur.fetchall())
    #remove project id
    features = rows[:, :-1]
    # load model and data in
    bst = xgb.Booster(model_file='xgb.model')

    dtest = xgb.DMatrix(features)
    preds = bst.predict(dtest)
    projects = []
    for i, pred in enumerate( preds ):
        projects.append([pred, int(rows[i][-1])])

    projects.sort(key=lambda x: x[0], reverse=True)
    print(projects[:20])
    display(eli5.format_as_html(eli5.explain_prediction_xgboost(bst, rows[0][:-1]),  show_feature_values=True))


def get_db_data():
    cur.execute("""
    WITH pt AS (SELECT c.project_id,
          sum(p.value) AS pledged,
          count(DISTINCT c.id) AS total_contributions
   FROM ((contributions c
          JOIN projects ON ((c.project_id = projects.id)))
         JOIN payments p ON ((p.contribution_id = c.id)))
   WHERE CASE
             WHEN ((projects.state)::text <> ALL (array['failed'::text,
                                                        'rejected'::text])) THEN (p.state = 'paid'::text)
             ELSE (p.state = ANY (confirmed_states()))
         END
     AND c.created_at <= projects.online_at + '3 days'::INTERVAL
   GROUP BY c.project_id,
            projects.id)
  (SELECT 
     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = p.category_id) AS category_count,
     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.mode = p.mode) AS mode_count,
 (coalesce(
            (SELECT state_id
             FROM cities
             WHERE cities.id = p.city_id
             LIMIT 1), 0) = coalesce(
            (SELECT state_id
             FROM addresses a
             WHERE a.id = u.address_id
               AND state_id IS NOT NULL
             LIMIT 1), 0))::integer AS same_state,
    coalesce(p.recommended, false)::integer as recommended,
    coalesce(p.online_days, 60)::integer AS online_days,
    coalesce(p.goal::float, 0) AS goal,
    coalesce(char_length(p.budget), 0) AS budget_length,
    coalesce(char_length(p.about_html), 0) AS about_length,
    coalesce(pt.pledged::float, 0) AS pledged,
    coalesce(pt.total_contributions::integer, 0) AS total_contributions,
    (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft')::integer as project_count,
    (SELECT count(*) from rewards r where r.project_id = p.id)::integer as reward_count,
    1 as y
   FROM contributions c
   JOIN users u ON u.id = c.user_id
   JOIN projects p ON p.id = c.project_id
   LEFT JOIN pt ON pt.project_id = p.id
   WHERE p.state NOT IN ('draft', 'rejected', 'deleted')
   LIMIT 220000)
union all
  (SELECT 
     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = p.category_id) AS category_count,
     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.mode = p.mode) AS mode_count,
 (coalesce(
            (SELECT state_id
             FROM cities
             WHERE cities.id = p.city_id
             LIMIT 1), 0) = coalesce(
            (SELECT state_id
             FROM addresses a
             WHERE a.id = u.address_id
               AND state_id IS NOT NULL
             LIMIT 1), 0))::integer AS same_state,
    coalesce(p.recommended, false)::integer as recommended,
    coalesce(p.online_days, 60)::integer AS online_days,
    coalesce(p.goal::float, 0) AS goal,
    coalesce(char_length(p.budget), 0) AS budget_length,
    coalesce(char_length(p.about_html), 0) AS about_length,
    coalesce(pt.pledged::float, 0) AS pledged,
    coalesce(pt.total_contributions::integer, 0) AS total_contributions,
    (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft')::integer as project_count,
    (SELECT count(*) from rewards r where r.project_id = p.id)::integer as reward_count,
    0 as y
from users u
join projects p on p.id = u.id / (((SELECT count(*) from users) / (SELECT count(*) from projects)) * 4)
join contributions c on c.user_id = u.id
LEFT JOIN pt ON pt.project_id = p.id
WHERE p.state NOT IN ('draft', 'rejected', 'deleted')
AND c.project_id != p.id
limit 220000)
    """)
    return np.array( cur.fetchall() )

def train_final_model(cache=False):
    if not cache:
        rows = get_db_data()
        features = encode_features(rows, 4)
        positive_ys = np.ones(220000, dtype = np.int)
        negative_ys = np.zeros(220000, dtype = np.int)
        ys = np.concatenate([positive_ys, negative_ys])
        try:
            dump_svmlight_file(features , ys, 'catarse.txt.all')
        except Exception as inst:
            print(inst)
        dtrain = xgb.DMatrix(features, label = ys)
    else:
        # load file from text file, also binary buffer generated by xgboost
        dtrain = xgb.DMatrix('catarse.txt.all')

    # specify parameters via map, definition are same as c++ version
    param = {'max_depth':4, 'eta':0.05, 'silent':1, 'booster': 'gbtree', 'min_child_weight': 3, 'objective':'binary:logistic', 'eval_metric': 'auc'}

    # specify validations set to watch performance
    watchlist = [(dtrain, 'train')]
    num_round = 2200
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=40)

    print(bst.get_fscore())
    print('best ite:', bst.best_iteration)
    print('best score:', bst.best_score)
    print('best ntree:', bst.best_ntree_limit)

    # dump model
    bst.dump_model('dump.raw.txt')

    # save model
    bst.save_model('xgb_final.model')
    plt.show(xgb.plot_importance(bst))

def train_model(cache=False):
    if not cache:
        rows = get_db_data()
        features = rows[:, :-1]
        ys = rows[:, -1]
        seed = 1
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(features, ys, test_size=test_size, random_state=seed)
        try:
            dump_svmlight_file(features , ys, 'catarse.txt.all')
            dump_svmlight_file(X_train, y_train, 'catarse.txt.train')
            dump_svmlight_file(X_test, y_test, 'catarse.txt.test')
        except Exception as inst:
            print(inst)

        dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = f_names)
        dtest = xgb.DMatrix(X_test, label = y_test, feature_names = f_names)
    else:
        # load file from text file, also binary buffer generated by xgboost
        dtrain = xgb.DMatrix('catarse.txt.train', feature_names = f_names)
        dtest = xgb.DMatrix('catarse.txt.test', feature_names = f_names)

    # specify parameters via map, definition are same as c++ version
    param = {'max_depth':5, 'eta':0.01, 'silent':1, 'booster': 'gbtree', 'min_child_weight': 2, 'objective':'binary:logistic', 'eval_metric': 'logloss'}

    # specify validations set to watch performance
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 8000
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30)
    bst.feature_names = f_names

    print(bst.get_fscore())
    print('best ite:', bst.best_iteration)
    print('best score:', bst.best_score)
    print('best ntree:', bst.best_ntree_limit)

    # this is prediction
    preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    labels = dtest.get_label()
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
    with open('pred.txt', 'w') as predictions:
        for item in preds:
            predictions.write("%s\n" % str( item ))

    # dump model
    bst.dump_model('dump.raw.txt')
    # dump model with feature map
    # bst.dump_model('dump.nice.txt', '../data/featmap.txt')

    # save dmatrix into binary buffer
    dtest.save_binary('dtest.buffer')
    dtrain.save_binary('dtrain.buffer')
    # save model
    bst.save_model('xgb.model')
    plt.show(xgb.plot_importance(bst))


def cv(cache=False):
    if not cache:
        rows = get_db_data()
        features = encode_features(rows, 4)
        positive_ys = np.ones(220000, dtype = np.int)
        negative_ys = np.zeros(220000, dtype = np.int)
        ys = np.concatenate([positive_ys, negative_ys])
        try:
            dump_svmlight_file(features , ys, 'catarse.txt.all')
        except Exception as inst:
            print(inst)
        dtrain = xgb.DMatrix(features, label = ys)
        data = load_svmlight_file("catarse.txt.all")
        X = data[0]
        y = data[1]
    else:
        # load file from text file, also binary buffer generated by xgboost
        dtrain = xgb.DMatrix('catarse.txt.all')
        data = load_svmlight_file('catarse.txt.all')
        X = data[0]
        y = data[1]

    predictors = X
    xgb1 = XGBClassifier(
        learning_rate =0.05,
        n_estimators=2000,
        max_depth=5,
        objective= 'binary:logistic',
        seed=27)

    xgb_param = xgb1.get_xgb_params()
    cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
                        metrics='auc', early_stopping_rounds=30)
    print(cvresult)
    xgb1.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    xgb1.fit(X, y, eval_metric='auc')
    #Predict training set:
    dtrain_predictions = xgb1.predict(X)
    # dtrain_predprob = xgb1.predict_proba(dtrain[predictors])[:,1]
    # #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    feat_imp = pandas.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    # specify parameters via map, definition are same as c++ version
    # param = {'max_depth': 4, 'eta':0.1, 'silent':1, 'booster': 'gbtree', 'min_child_weight': 3, 'objective':'binary:logistic'}

    # num_round = 800
    # bst = xgb.cv(param, dtrain, num_round, nfold=10,
    #              metrics={'auc'}, seed=123, early_stopping_rounds=10,
    #              verbose_eval=1,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

    # print(bst)


get_predictions(233124)
# train_model(cache=False)
# train_final_model(cache=False)
# cv(cache=False)

cur.close()
conn.close()

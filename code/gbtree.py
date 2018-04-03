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
        'cat0',
        'cat1',
        'cat4',
        'cat7',
        'cat9',
        'cat10',
        'cat12',
        'cat13',
        'cat14',
        'cat15',
        'cat16',
        'cat17',
        'cat18',
        'cat19',
        'cat20',
        'cat21',
        'cat23',
        'cat31',
        'cat35',
        'aon',
        'flex',
        'sub',
        'state0',
        'state1',
        'state2',
        'state3',
        'state4',
        'state5',
        'state6',
        'state7',
        'state8',
        'state9',
        'state10',
        'state11',
        'state12',
        'state13',
        'state14',
        'state15',
        'state16',
        'state17',
        'state18',
        'state19',
        'state20',
        'state21',
        'state22',
        'state23',
        'state24',
        'state25',
        'state26',
        'state27',
        'ustate0',
        'ustate1',
        'ustate2',
        'ustate3',
        'ustate4',
        'ustate5',
        'ustate6',
        'ustate7',
        'ustate8',
        'ustate9',
        'ustate10',
        'ustate11',
        'ustate12',
        'ustate13',
        'ustate14',
        'ustate15',
        'ustate16',
        'ustate17',
        'ustate18',
        'ustate19',
        'ustate20',
        'ustate21',
        'ustate22',
        'ustate23',
        'ustate24',
        'ustate25',
        'ustate26',
        'ustate27',
        'ucat0',
        'ucat1',
        'ucat4',
        'ucat7',
        'ucat9',
        'ucat10',
        'ucat12',
        'ucat13',
        'ucat14',
        'ucat15',
        'ucat16',
        'ucat17',
        'ucat18',
        'ucat19',
        'ucat20',
        'ucat21',
        'ucat23',
        'ucat31',
        'ucat35',
        'recommended',
        'online_days',
        'goal',
        'budget',
        'description',
        'pledged',
        'contributions',
        'owner_projects'
    ]

def get_categories():
    cur.execute("""select id from categories order by id;""")
    rows = np.array(cur.fetchall())
    return rows

def get_modes():
    cur.execute("""select distinct mode from projects order by mode;""")
    rows = np.array(cur.fetchall())
    return rows

def get_states():
    cur.execute("""select id from states order by id;""")
    rows = np.array(cur.fetchall())
    return rows

def encode_features(data, n_categories):
    # get all possible values for categorical data so that features stay independent of data variety
    categories = get_categories()
    states = get_states()
    modes = get_modes()
    states = np.concatenate((states, [ [0] ]))
    categories = np.concatenate((categories, [ [0] ]))

    first_row = data[0]
    for id in categories:
        new_row = first_row
        new_row[0] = id[0]
        data = np.concatenate((data, [ new_row ]))
    for id in modes:
        new_row = first_row
        new_row[1] = id[0]
        data = np.concatenate((data, [ new_row ]))
    for id in states:
        new_row = first_row
        new_row[2] = id[0]
        data = np.concatenate((data, [ new_row ]))
    for id in states:
        new_row = first_row
        new_row[3] = id[0]
        data = np.concatenate((data, [ new_row ]))

    encoded_x = None
    for i in range(0, n_categories):
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform([item[i] for item in data])
        feature = feature.reshape(data.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        feature = onehot_encoder.fit_transform(feature)
        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1)

    to_remove = len(categories) + len(modes) + 2 * len(states)
    formated_data = np.concatenate((encoded_x[:-to_remove], [item[n_categories:] for item in data[:-to_remove]]), axis=1)
    return formated_data.astype(float)


def get_predictions(user_id):
    cur.execute("""
        SELECT
    --categorical values, need to one-hot encode
    p.category_id,
    p.mode,
    COALESCE((SELECT state_id from cities where cities.id = p.city_id limit 1), 0) as project_state,
    COALESCE(u.state_id, 0) as user_state,
    --integer values
    u.c1,
    u.c4,
    u.c7,
    u.c9,
    u.c10,
    u.c12,
    u.c13,
    u.c14,
    u.c15,
    u.c16,
    u.c17,
    u.c18,
    u.c19,
    u.c20,
    u.c21,
    u.c23,
    u.c31,
    u.c35,
    COALESCE(p.recommended, false)::integer,
    COALESCE(p.online_days::integer, 0),
    COALESCE(p.goal::integer, 0),
    COALESCE( char_length(p.budget), 0 ),
    COALESCE( char_length(p.about_html), 0 ),
    COALESCE((SELECT sum(pa.value) from contributions c join payments pa on pa.contribution_id = c.id where pa.state IN ('paid', 'pending') and c.project_id = p.id and pa.created_at <= p.created_at + '3 days'::interval)::float, 0) as pledged,
    (SELECT count(*) from contributions c where c.project_id = p.id and c.created_at <= p.created_at + '3 days'::interval) as contributions_total,
    (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft') as project_count,
    p.id
    from projects p
    LEFT JOIN LATERAL (
    SELECT
    u.id,
    a.state_id as state_id,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 1 and contributions.was_confirmed) as c1,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 4 and contributions.was_confirmed) as c4,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 7 and contributions.was_confirmed) as c7,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 9 and contributions.was_confirmed) as c9,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 10 and contributions.was_confirmed) as c10,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 12 and contributions.was_confirmed) as c12,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 13 and contributions.was_confirmed) as c13,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 14 and contributions.was_confirmed) as c14,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 15 and contributions.was_confirmed) as c15,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 16 and contributions.was_confirmed) as c16,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 17 and contributions.was_confirmed) as c17,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 18 and contributions.was_confirmed) as c18,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 19 and contributions.was_confirmed) as c19,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 20 and contributions.was_confirmed) as c20,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 21 and contributions.was_confirmed) as c21,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 23 and contributions.was_confirmed) as c23,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 31 and contributions.was_confirmed) as c31,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 35 and contributions.was_confirmed) as c35
    from users u
    left join addresses a on a.id = u.address_id
    WHERE u.id = """ + str( user_id ) + """
    ) u on true
    where p.state = 'online'
    AND NOT EXISTS (select true from contributions c2 where c2.project_id = p.id and c2.user_id = u.id)
    --and p.category_id = 14
   and p.id IN(56252)
    """)

    rows = cur.fetchall()
    rows = np.array( rows )

    features = encode_features(rows, 4)
    features_copy = list(features)
    for i, feat in enumerate(features_copy):
        feat = feat[:-1]
        features_copy[i] = feat

    # load model and data in
    features_copy = np.array(features_copy)
    bst = xgb.Booster(model_file='xgb.model')
    
    dtest = xgb.DMatrix(features_copy)
    print(features_copy)
    preds = bst.predict(dtest)
    projects = []
    for i, pred in enumerate( preds ):
        projects.append([pred, features[i][-1]])

    projects.sort(key=lambda x: x[0], reverse=True)
    print(projects)
    display(eli5.format_as_html(eli5.explain_prediction_xgboost(bst, features_copy[0]),  show_feature_values=True))


def get_db_data():
    cur.execute("""
    --get pledges from the first 3 days
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

  (SELECT --categorical values, need to one-hot encode
 p.category_id,
 p.mode,
 coalesce(
            (SELECT state_id
             FROM cities
             WHERE cities.id = p.city_id
             LIMIT 1), 0) AS project_state,
 coalesce(
            (SELECT state_id
             FROM addresses a
             WHERE a.id = u.address_id
               AND state_id IS NOT NULL
             LIMIT 1), 0) AS user_state, --integer values

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 1) AS c1,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 4) AS c4,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 7) AS c7,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 9) AS c9,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 10) AS c10,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 12) AS c12,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 13) AS c13,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 14) AS c14,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 15) AS c15,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 16) AS c16,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 17) AS c17,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 18) AS c18,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 19) AS c19,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 20) AS c20,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 21) AS c21,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 23) AS c23,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 31) AS c31,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 35) AS c35,
    coalesce(p.recommended, false)::integer,
    coalesce(p.online_days, 60)::integer AS online_days,
    coalesce(p.goal::float, 0) AS goal,
    coalesce(char_length(p.budget), 0) AS budget_length,
    coalesce(char_length(p.about_html), 0) AS about_length,
    coalesce(pt.pledged::float, 0) AS pledged,
    coalesce(pt.total_contributions::integer, 0) AS total_contributions,
    (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft')::integer as project_count
   FROM contributions c
   JOIN users u ON u.id = c.user_id
   JOIN projects p ON p.id = c.project_id
   LEFT JOIN pt ON pt.project_id = p.id
   WHERE p.state NOT IN ('draft', 'rejected', 'deleted')
   LIMIT 220000)
UNION ALL
--negative examples

    (select  p.category_id,
 p.mode,
 coalesce(
            (SELECT state_id
             FROM cities
             WHERE cities.id = p.city_id
             LIMIT 1), 0) AS project_state,
 coalesce(
            (SELECT state_id
             FROM addresses a
             WHERE a.id = u.address_id
               AND state_id IS NOT NULL
             LIMIT 1), 0) AS user_state, --integer values

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 1) AS c1,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 4) AS c4,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 7) AS c7,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 9) AS c9,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 10) AS c10,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 12) AS c12,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 13) AS c13,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 14) AS c14,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 15) AS c15,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 16) AS c16,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 17) AS c17,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 18) AS c18,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 19) AS c19,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 20) AS c20,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 21) AS c21,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 23) AS c23,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 31) AS c31,

     (SELECT count(*)
      FROM contributions
      JOIN projects ON projects.id = contributions.project_id
      WHERE contributions.user_id = u.id
        AND projects.category_id = 35) AS c35,
 coalesce(p.recommended, false)::integer,
 coalesce(p.online_days, 60)::integer AS online_days,
 coalesce(p.goal, 0)::float AS goal,
 coalesce(char_length(p.budget), 0) AS budget_length,
 coalesce(char_length(p.about_html), 0) AS about_length,
 coalesce(pt.pledged, 0)::float AS pledged,
 coalesce(pt.total_contributions, 0)::integer AS total_contributions,
 (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft') as project_count
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

        features = encode_features(rows, 4)
        positive_ys = np.ones(220000, dtype = np.int)
        negative_ys = np.zeros(220000, dtype = np.int)
        ys = np.concatenate([positive_ys, negative_ys])

        seed = 5
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
# train_model(cache=True)
# train_final_model(cache=False)
# cv(cache=False)

cur.close()
conn.close()

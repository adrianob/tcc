#!/usr/bin/python
import numpy as np
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

# np.set_printoptions(threshold=np.inf)

try:
    conn = psycopg2.connect(dbname='catarse_db', user='catarse', host='localhost', password='example', port='5445')
except:
    print(  "I am unable to connect to the database" )

cur = conn.cursor()

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
    formated_data = np.concatenate((encoded_x, [item[n_categories:] for item in data]), axis=1)
    return formated_data[:-to_remove]

def get_predictions(user_id):
    cur.execute("""
    WITH pt AS
  (SELECT c.project_id,
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
    COALESCE(p.recommended, false),
    COALESCE(p.online_days, 0),
    COALESCE(p.goal, 0),
    COALESCE( char_length(p.budget) ),
    COALESCE( char_length(p.about_html) ),
    COALESCE( pt.pledged, 0 ), --@TODO get from first 3 days
    COALESCE( pt.total_contributions, 0),
    (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft') as project_count,
    p.id
    from projects p
    LEFT join pt on pt.project_id = p.id
    LEFT JOIN LATERAL (
    SELECT
    u.id,
    a.state_id as state_id,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 1) as c1,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 4) as c4,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 7) as c7,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 9) as c9,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 10) as c10,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 12) as c12,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 13) as c13,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 14) as c14,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 15) as c15,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 16) as c16,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 17) as c17,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 18) as c18,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 19) as c19,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 20) as c20,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 21) as c21,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 23) as c23,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 31) as c31,
    (select count(*) from contributions join projects on projects.id = contributions.project_id where contributions.user_id = u.id and projects.category_id = 35) as c35
    from users u
    left join addresses a on a.id = u.address_id
    WHERE u.id = """ + str( user_id ) + """
    ) u on true
    where p.state = 'online' """)

    rows = cur.fetchall()
    rows = np.array( rows )

    features = encode_features(rows, 4)
    # load model and data in
    bst2 = xgb.Booster(model_file='xgb.model')
    dtest2 = xgb.DMatrix(features[:-1])
    preds2 = bst2.predict(dtest2)
    projects = []
    for i, pred in enumerate( preds2 ):
        projects.append([pred, features[i][-1]])

    projects.sort(key=lambda x: x[0], reverse=True)
    print(projects)

def get_db_data():
    cur.execute("""
    WITH pt AS
  (SELECT c.project_id,
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
 coalesce(p.recommended, false),
 coalesce(p.online_days, 60) AS online_days,
 coalesce(p.goal, 0) AS goal,
 coalesce(char_length(p.budget), 0) AS budget_length,
 coalesce(char_length(p.about_html), 0) AS about_length,
 coalesce(pt.pledged, 0) AS pledged,
 coalesce(pt.total_contributions, 0) AS total_contributions,
 (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft') as project_count
   FROM contributions c
   JOIN users u ON u.id = c.user_id
   JOIN projects p ON p.id = c.project_id
   LEFT JOIN pt ON pt.project_id = p.id
   WHERE p.state NOT IN ('draft', 'rejected', 'deleted')
   LIMIT 320000)
UNION ALL
--negative examples
  (SELECT --categorical values, need to one-hot encode
 p.category_id,
 p.mode,
 COALESCE(
            (SELECT state_id
             FROM cities
             WHERE cities.id = p.city_id
             LIMIT 1), 0) AS project_state,
 COALESCE(u.state_id, 0) AS user_state, --integer values
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
 COALESCE(p.recommended, false),
 COALESCE(p.online_days, 0)AS online_day,
 COALESCE(p.goal, 0) AS goal,
 COALESCE(char_length(p.budget), 0) AS budget_length,
 COALESCE(char_length(p.about_html), 0) AS about_length,
 COALESCE(pt.pledged, 0) AS pledged,
 COALESCE(pt.total_contributions, 0) AS total_contributions,
 (SELECT count(*) from projects where projects.user_id = p.user_id AND projects.state != 'draft') as project_count
   FROM projects p
   LEFT JOIN LATERAL
     ( SELECT u.id,
              COALESCE(
                         (SELECT state_id
                          FROM addresses a
                          WHERE a.id = u.address_id
                            AND state_id IS NOT NULL
                          LIMIT 1), 0) AS state_id,

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
           AND projects.category_id = 35) AS c35
      FROM users u
      JOIN contributions con ON con.user_id = u.id
      WHERE con.project_id <> p.id ) u ON TRUE
   LEFT JOIN pt ON pt.project_id = p.id
   WHERE p.state NOT IN ('draft', 'rejected', 'deleted')
     AND NOT EXISTS
       (SELECT TRUE
        FROM contributions c
        WHERE c.project_id = p.id
          AND c.user_id = u.id)
     AND EXISTS
       (SELECT TRUE
        FROM contributions c
        WHERE c.user_id = u.id)
   LIMIT 320000 )
    """)
    return np.array( cur.fetchall() )

def train_model(cache=False):
    if not cache:
        rows = get_db_data()

        features = encode_features(rows, 4)
        positive_ys = np.ones(320000, dtype = np.int)
        negative_ys = np.zeros(320000, dtype = np.int)
        ys = np.concatenate([positive_ys, negative_ys])

        seed = 4
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(features, ys, test_size=test_size, random_state=seed)
        try:
            dump_svmlight_file(X_train, y_train, 'catarse.txt.train')
            dump_svmlight_file(X_test, y_test, 'catarse.txt.test')
        except Exception as inst:
            print(inst)
        dtrain = xgb.DMatrix(X_train, label = y_train)
        dtest = xgb.DMatrix(X_test, label = y_test)
    else:
        # load file from text file, also binary buffer generated by xgboost
        dtrain = xgb.DMatrix('catarse.txt.train')
        dtest = xgb.DMatrix('catarse.txt.test')

    # specify parameters via map, definition are same as c++ version
    param = {'max_depth':4, 'eta':0.2, 'silent':1, 'booster': 'gbtree', 'min_child_weight': 1, 'objective':'binary:logistic'}

    # specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 1000
    bst = xgb.train(param, dtrain, num_round, watchlist)

    print(bst.get_fscore())

    # this is prediction
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
    with open('pred.txt', 'w') as predictions:
        for item in preds:
            predictions.write("%s\n" % str( item ))

    # dump model
    bst.dump_model('dump.raw.txt')
    # dump model with feature map
    bst.dump_model('dump.nice.txt', '../data/featmap.txt')

    # save dmatrix into binary buffer
    dtest.save_binary('dtest.buffer')
    # save model
    bst.save_model('xgb.model')

    # predictions = [round(value) for value in preds]
    # # evaluate predictions
    # accuracy = accuracy_score(dtest, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))


get_predictions(90198)
# train_model()

cur.close()
conn.close()

# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_trend(dataset, min_thres=None, max_thres=None):
  csv, outcome_col, intuition_attr = dataset['csv'], dataset['outcome_col'], dataset['int_attr']
  if csv.endswith('xlsx'):
    csv_data = pd.read_excel(csv)
  else:
    csv_data = pd.read_csv(csv)
  csv_data.fillna(csv_data.mean(), inplace=True)
  X, Y = csv_data.drop(outcome_col, axis=1), csv_data[outcome_col]
  min_thres = int(X.min(axis=0)[intuition_attr]) if not min_thres else min_thres
  max_thres = int(X.max(axis=0)[intuition_attr]) if not max_thres else max_thres

  scaler = MinMaxScaler()
  #clf = LogisticRegression(solver='lbfgs')
  clf = RandomForestClassifier()
  #clf = SVC(probability=True)
  #clf = SVC(probability=True, kernel='poly', degree=2)
  #clf = MLPClassifier((15,))
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
  pipeline = make_pipeline(scaler, clf)

  pipeline.fit(X_train, Y_train)
  Y_pred = pipeline.predict(X_test)
  print("Accuracy = {:.2f}".format(accuracy_score(Y_test, Y_pred)))

  sample = X.sample(n=1).copy()

  t = np.arange(min_thres, max_thres, 0.1).tolist()
  test_df = pd.DataFrame(data=np.repeat(sample.values, len(t), axis=0), columns=X.columns)
  test_df[intuition_attr] = t

  preds = pipeline.predict_proba(test_df)[:, 1]
  plt.scatter(t, preds)
  plt.ylabel('outcome prob')
  plt.xlabel('intuition attribute')
  plt.savefig('plots/trend.png')
  plt.close()

if __name__ == "__main__":
  dataset = {
    'heart_attack': {
      'csv': './data/heart_attack.csv',
      'outcome_col': 'num',
      'int_attr': 'chol'
    }
  }
  dataset_type = 'ibm_deal_data'
  plot_trend(dataset[dataset_type])
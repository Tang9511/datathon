# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from collections import Counter

class SBBTree():
	"""Stacking,Bootstap,Bagging----SBBTree"""
	""" author: Cookly """
	""" modifier: Johncole """
	def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
		"""
		Initializes the SBBTree.
        Args:
          params : lgb params.
          stacking_num : k_flod stacking.
          bagging_num : bootstrap num.
          bagging_test_size : bootstrap sample rate.
          num_boost_round : boost num.
		  early_stopping_rounds : early_stopping_rounds.
        """
		self.params = params
		self.stacking_num = stacking_num
		self.bagging_num = bagging_num
		self.bagging_test_size = bagging_test_size
		self.num_boost_round = num_boost_round
		self.early_stopping_rounds = early_stopping_rounds

		self.model = lgb
		self.stacking_model = []
		self.bagging_model = []

	def fit(self, X, y):
		""" fit model. """
		if self.stacking_num > 1:
			layer_train = np.zeros((X.shape[0], 1))
			# cross validation
			self.SK = KFold(n_splits=self.stacking_num, shuffle=True, random_state=34)
			for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
				X_train = X[train_index]
				y_train = y[train_index]
				X_test = X[test_index]
				y_test = y[test_index]

				lgb_train = lgb.Dataset(X_train, y_train)
				lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

				gbm = lgb.train(self.params,
							lgb_train,
							num_boost_round=self.num_boost_round,
							valid_sets=lgb_eval,
							early_stopping_rounds=self.early_stopping_rounds)

				self.stacking_model.append(gbm)

				pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
				if self.params['objective'] == 'regression':
					layer_train[test_index, 0] = pred_y
				elif self.params['objective'] == 'multiclass':
					layer_train[test_index, 0] = np.argmax(pred_y, axis=1)

			X = np.hstack((X, layer_train[:, 0].reshape((-1, 1))))

		for bn in range(self.bagging_num):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)
	
			lgb_train = lgb.Dataset(X_train, y_train)
			lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

			gbm = lgb.train(self.params,
							lgb_train,
							num_boost_round=10000,
							valid_sets=lgb_eval,
							early_stopping_rounds=200)

			self.bagging_model.append(gbm)
		
	def predict(self, X_pred):
		""" predict test data. """
		if self.stacking_num > 1:
			test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
			for sn, gbm in enumerate(self.stacking_model):
				pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
				if self.params['objective'] == 'regression':
					test_pred[:, sn] = pred
				elif self.params['objective'] == 'multiclass':
					test_pred[:, sn] = np.argmax(pred, axis=1)

			if self.params['objective'] == 'regression':
				X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1,1))))
			elif self.params['objective'] == 'multiclass':
				X_pred = np.hstack((X_pred, test_pred))


		test_pred = np.zeros((X_pred.shape[0], self.bagging_num))
		for bn, gbm in enumerate(self.bagging_model):
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			if self.params['objective'] == 'regression':
				test_pred[:, bn] = pred
			elif self.params['objective'] == 'multiclass':
				test_pred[:, bn] = np.argmax(pred, axis=1)
		#averaging
		if self.params['objective'] == 'regression':
			test_pred = test_pred.mean(axis=1).reshape((-1, 1))
		# majority voting
		elif self.params['objective'] == 'multiclass':
			tp = []
			for i in range(test_pred.shape[0]):
				c = Counter(test_pred[i, :])
				tp.append(int(c.most_common()[0][0]))
			test_pred = np.array(tp).reshape((-1, 1))
		return test_pred




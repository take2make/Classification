import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import two_sin
import classificator
from sklearn.model_selection import train_test_split

class Model():
	def __init__(self):
		self.model = classificator.Net(5)
		self.loss = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), 
		                             lr=1.0e-3)

		data, target = two_sin(N=512*2, rotation=np.pi)
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

		self.X_train = torch.FloatTensor(X_train)
		self.X_test = torch.FloatTensor(X_test)
		self.y_train = torch.LongTensor(y_train)
		self.y_test = torch.LongTensor(y_test)

	def train(self):
		batch_size = 10
		for epoch in range(100):
			order = np.random.permutation(len(self.X_train))
			for start_index in range(0, len(self.X_train), batch_size):
				self.optimizer.zero_grad()

				batch_indexes = order[start_index:start_index+batch_size]

				x_batch = self.X_train[batch_indexes]
				y_batch = self.y_train[batch_indexes]

				preds = self.model.forward(x_batch) 

				loss_value = self.loss(preds, y_batch)
				loss_value.backward()

				self.optimizer.step()
			if epoch % 100 == 0:
				test_preds = self.model.predict_proba(self.X_test)
				test_preds = test_preds.argmax(dim=1)
				print((test_preds == self.y_test).float().mean())
	def _plot_separation_curve_(self):
		step = 0.02

		x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
		y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1

		xx, yy =  torch.meshgrid(torch.arange(x_min, x_max, step),
		                     torch.arange(y_min, y_max, step))

		preds = self.model.predict_proba(torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1))
		preds_class = preds.data.numpy().max(axis=1)
		preds_class = preds_class.reshape(xx.shape)

		plt.pcolormesh(xx, yy, -preds_class, cmap=plt.get_cmap('copper'))

	def _plot_points_(self, n_classes=2):
		colors = ['lightskyblue', 'aliceblue']
		for i, color in zip(range(n_classes), colors):
			indexes = np.where(self.y_train==i)
			plt.scatter(self.X_train[indexes, 0],
				self.X_train[indexes, 1],
				c = color)
	def show(self):
		plt.figure(figsize=(12,10), dpi=400)
		self._plot_separation_curve_()
		self._plot_points_()
		plt.show()


if __name__=="__main__":
	model = Model()
	model.train()
	model.show()

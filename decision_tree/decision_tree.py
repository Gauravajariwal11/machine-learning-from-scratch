#Name- Gaurav Ajariwal
#UTA ID- 1001396273

import sys
import numpy as np 
import random
import math



class PreprocessData():

	def __init__(self, train_file, test_file):
		self.train_file = train_file
		self.test_file = test_file
		self.load_data(self.train_file, True)
		self.load_data(self.test_file, False)

	def load_data(self, data_file, is_train):
		data = np.loadtxt(data_file)
		
		if is_train:
			self.train_data=data
			self.attr = data.shape[1] - 1
			unique_data = sorted(np.unique(data[: ,[-1]]))
			self.num_class = len(unique_data)
			self.mapping = {}
			self.idx = {}


			for i, j in enumerate(unique_data):
				self.mapping[i] = j
				self.idx[j] = i

		else:
			self.test_data = data

class Forest():
    
	def __init__(self, num_trees, option, prune, data):

		self.data = data
		self.num_trees = num_trees
		self.trees = [DecisionTree(i, option, prune, data).DTL_TopLevel() for i in range(1,self.num_trees+1)]

	def classify(self, data):
		forest_dist = []

		for i in self.trees:
			dist = i.predict(data)
			forest_dist.append(dist)

		forest_dist = np.array(forest_dist)
		return self.data.mapping[int(np.argmax(np.mean(forest_dist, axis = 0)))]

	def predictions(self):
		object_id = 1
		count = 0
		curr_val = 0
		for i in self.data.test_data:
			predicted_cal = int(self.classify(i[:-1]))
			true_val = int(i[-1])
			if (predicted_cal != true_val):
				accuracy=0
			else:
				accuracy=1            
			curr_val += accuracy
			count += 1
			print(f"ID={object_id:5d}, predicted={predicted_cal:3d}, true={true_val:3d}, accuracy={accuracy:4.2f}\n")
			object_id += 1
		print(f"classification accuracy={curr_val/count:6.4f}")


class Tree:
	def __init__(self, id, tree_gain = None, tree_attr = None, tree_dist = None, threshold_tree = None, is_leaf = False):
		self.id = id
		self.tree_gain = tree_gain
		self.tree_attr = tree_attr
		self.tree_dist = tree_dist
		self.threshold_tree = threshold_tree
		self.is_leaf = is_leaf
		self.left = None
		self.right = None

	def predict(self, data):
		if self.is_leaf:
			return self.tree_dist
		else:
			if data[self.tree_attr] < self.threshold_tree:
				return self.left.predict(data)
			else:
				return self.right.predict(data)
	def print(self):
		queue = []
		queue.append((self, 1))
		while queue:
			count = len(queue)
			while count > 0:
				temp = queue.pop(0)
				if temp[0].is_leaf:
					feature = -1
					thr = -1
					gain = 0
				else:
					feature = temp[0].tree_attr + 1
					thr = temp[0].threshold_tree
					gain = temp[0].tree_gain
				if temp[0].left:
					queue.append((temp[0].left, 2 * temp[1]))
				if temp[0].right:
					queue.append((temp[0].right, 2 * temp[1] + 1))
				count -= 1
				print(f"tree={temp[0].id:2d}, node={temp[1]:3d}, feature={feature:2d}, thr={thr:6.2f}, gain={gain}\n")



class DecisionTree:
	def __init__(self, id_, option, prune, data):
		self.data = data
		self.id = id_
		self.option = option
		self.prune = int(prune)
		self.idx = data.idx
		self.mapping = data.mapping
		self.attr = data.attr
		self.num_class = data.num_class

	def if_equal(self, examples):
		values = examples[: ,[-1]]
		element = examples[: ,[-1]][0]
		for i in values:
			if i != element:
				return False
		return True

	def DTL_TopLevel(self):
		default = self.distribution(self.data.train_data)
		if self.option == "randomized":
			choose_attribute = self.random
		elif self.option == "optimized":
			choose_attribute = self.optimized
		return self.DTL(self.data.train_data, default, choose_attribute)

	def DTL(self, examples, default, choose_attribute):
		if examples.shape[0] == 0 or examples.shape[0] < self.prune:
			return Tree(self.id, is_leaf = True, tree_dist = default)

		elif self.if_equal(examples):
			return Tree(self.id, is_leaf = True, tree_dist = default)
		
		else:
			best_gain, best_attr, best_threshold = choose_attribute(examples)

			tree = Tree(self.id,tree_attr = best_attr, threshold_tree = best_threshold, tree_gain = best_gain)

			left = []
			right = []

			for i in examples:
				idx = best_attr
				if i[idx] < best_threshold:
					left.append(i)
				elif i[idx] >= best_threshold:
					right.append(i)

			tree.left = self.DTL(np.array(left), self.distribution(examples), choose_attribute)
			tree.right = self.DTL(np.array(right), self.distribution(examples), choose_attribute)
			return tree

	def optimized(self, examples):
		max_gain=-1 
		max_attribute=-1 
		max_threshold=-1
		
		for i in range(self.attr):
			values = examples[: ,[i]]

			L = np.amin(values)
			M = np.amax(values)
			
			step = (M-L) / 51
			for k in range(1, 51):
				threshold = (L + k) * step
				gain = self.information_gain(examples, i, threshold)
				if gain > max_gain:
					max_gain = gain
					max_attribute = i
					max_threshold = threshold
		return max_gain, max_attribute, max_threshold

	def random(self, examples):

		max_gain=-1
		max_threshold=-1
		i = random.randint(0, self.attr - 1)
		values = examples[: ,[i]]
		L = np.amin(values)
		M = np.amax(values)
		step = (M-L) / 51
		for j in range(1, 51):
			threshold = (L + j) * step
			gain = self.information_gain(examples, i, threshold)
			if gain > max_gain:
				max_gain = gain
				max_threshold = threshold
		return max_gain, i, max_threshold

	def information_gain(self, examples, A, threshold):

		base_entropy = self.entropy(examples)
		N1 = []
		N2 = []
		for i in examples:
			if i[A] < threshold:
				N1.append(i)
			elif i[A] >= threshold:
				N2.append(i)

		N1=np.array(N1)
		N2=np.array(N2)

		base_entropy-=(N1.shape[0]/examples.shape[0])*self.entropy(N1)
		base_entropy-=(N2.shape[0]/examples.shape[0])*self.entropy(N2)

		return base_entropy

	def entropy(self, examples):

		count = {}
		for i in examples:
			label = int(i[-1])
			count[label] = count.get(label, 1) + 1
		return sum([-(j/examples.shape[0]) * np.log2(j/examples.shape[0]) for j in count.values()])

	def distribution(self, data):
		zeros = np.zeros(self.num_class)
		for i in data:

			label = int(i[-1])
			zeros[self.idx[label]] += 1
			zeros /= data.shape[0]

		return zeros

	def predictions(self, root):
		object_id = int(1)
		total = 0
		curr_val = 0
		for i in self.data.test_data:
			true_val = int(i[-1])
			dist = root.predict(i[:-1])
			predicted_val = int(np.argmax(dist))
			predicted_val = int(self.mapping[int(predicted_val)])
			
			total += 1

			if (true_val != predicted_val):
				accuracy=0
			else:
				accuracy=1
			curr_val += accuracy
			print(f"ID={object_id:5d}, predicted={predicted_val:3d}, true={true_val:3d}, accuracy={accuracy:4.2f}\n")
			object_id += 1
		print(f"classification accuracy={curr_val/total:6.4f}")

def main():
	data = PreprocessData(sys.argv[1], sys.argv[2])
	if sys.argv[3] == "forest3":
		forest = Forest(3,"randomized", sys.argv[4], data)
		for i in forest.trees:
			i.print()
		forest.predictions()
	elif sys.argv[3] == "forest15":
		forest = Forest(15,"randomized", sys.argv[4], data)
		for i in forest.trees:
			i.print()
		forest.predictions()
	else:
		tree = DecisionTree(1, sys.argv[3], sys.argv[4], data)
		root = tree.DTL_TopLevel()
		root.print()
		tree.predictions(root)

if __name__ == "__main__":
	main()

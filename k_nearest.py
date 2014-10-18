from numpy import array,tile

def create_data_set():
	group = array([[10, 11], [10, 10], [0, 2], [0, 1]])
	labels = ["A", "A", "B", "B"]
	return group, labels


group,labels = create_data_set()

def classify(test_data, group, labels, k):
	data_set_size = group.shape[0]
	diff_matrix = tile(test_data, [data_set_size, 1]) - group
	sq_diff_matrix = diff_matrix ** 2
	sq_distance = sq_diff_matrix.sum(axis=1)
	sorted_distances = sq_distance.argsort()
	class_count = {}
	for i in range(k):
		vote_label = labels[sorted_distances[i]]
		class_count[vote_label] = class_count.get(vote_label, 0) + 1
	sorted_class_count = sorted(class_count.iteritems(), key=lambda x:x[1], reverse=True)
	return sorted_class_count[0][0]


test_data = [1, 2]
k = 3
classify(test_data, group, labels, k)

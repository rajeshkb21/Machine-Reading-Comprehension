import os
import pickle
import random
import numpy as np
import math
import random
import torch
from torch import nn
from torch.nn import init
from torch import optim
from NNmodel import model
import time
from tqdm import tqdm
import pandas as pd


# Saves input python data structure as pickle file in project root
def save_file(file_name, data):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)     


# Opens pickle file and returns stored data object
def open_file(file_name):
	with open(file_name, 'rb') as handle:
		return pickle.load(handle)


def build_indices(information_dictionary, start_index):
	# From token to its index
	forward_dict = {'UNK': 0}
	# From index to token
	backward_dict = {0: 'UNK'}
	counts = {}
	i = 1
	num_entries = len(information_dictionary)
	for n in range(0, num_entries):
		entry = information_dictionary[start_index+n]
		question = entry['question']
		context = entry['context']
		words = question.split(' ') + context.split(' ')
		for word in words:
			if word not in forward_dict:
				forward_dict[word] = i
				backward_dict[i] = word
				counts[i] = 1
				i += 1
			else: 
				index = forward_dict[word]
				num_counts = counts[index]
				counts[index] = num_counts+1
	return forward_dict, backward_dict, counts


def read_input():
	pass


#converts training data into form where words are replaced with numbers from the forward_dictionary
def encode(training_dictionary, training_vocab_forward_dictionary, start_index=0):
	training_vectors = []
	num_training_vectors = len(training_dictionary)
	for i in range(0,num_training_vectors):
		entry = training_dictionary[i+start_index]
		question = entry['question']
		context = entry['context']
		question_words = question.split(' ')
		context_words = context.split(' ')
		question_symbols = encode_line(question_words, training_vocab_forward_dictionary)
		context_symbols = encode_line(context_words, training_vocab_forward_dictionary)
		vector = [question_symbols,context_symbols]
		training_vectors.append(vector)
	return training_vectors


def encode_line(word_list, training_vocab_forward_dict):
	num_words = len(word_list)
	for i in range(0, num_words):
		word = word_list[i]
		index = 0
		if word in training_vocab_forward_dict:
			index = training_vocab_forward_dict[word]
		word_list[i] = index
	return word_list


def introduce_unknown_words(training_vectors, counts):
	num_vectors = len(training_vectors)
	num_changes = 0
	for i in range(0,num_vectors):
		vector = training_vectors[i]
		question = vector[0]
		question_length = len(question)
		context = vector[1]
		context_length = len(context)
		for j in range(0,question_length):
			word_id = question[j]
			seed = random.random()
			if counts[word_id] <= 3 and seed < 0.25:
				question[j] = 0
				num_changes += 1
		for j in range(0,context_length):
			word_id = context[j]
			seed = random.random()
			if counts[word_id] <= 3 and seed < 0.25:
				context[j] = 0
				num_changes += 1
		training_vectors[i][0] = question
		training_vectors[i][1] = context
	return training_vectors, num_changes


def get_data_labels(training_dictionary, start_index=0):
	labels = []
	num_entries = len(training_dictionary)
	for i in range(0,num_entries):
		entry = training_dictionary[i+start_index]
		answer_not_found = entry['impossible_val']
		label = 1
		if answer_not_found:
			label = 0
		labels.append(label)
	return labels


def get_pred_ids(test_dictionary, start_index=0):
	num_test_vectors = len(test_dictionary)
	pred_ids = []
	for i in range(0,num_test_vectors):
		vector = test_dictionary[start_index+i]
		pred_id = vector['id']
		pred_ids.append(pred_id)
	return pred_ids

if __name__ == '__main__':
	# TRAINING_DICTIONARY = "training_dictionary.pickle"
	# DEVELOPMENT_DICTIONARY = "development_dictionary.pickle"
	# TEST_DICTIONARY = "test_dictionary.pickle"
	# TRAINING_VOCAB_FORWARD_DICT = "training_vocab_forward_dictionary.pickle"
	
	# training_dict = open_file(TRAINING_DICTIONARY)
	# development_dict = open_file(DEVELOPMENT_DICTIONARY)
	# test_dict = open_file(TEST_DICTIONARY)

	# training_vocab_forward_dict = open_file(TRAINING_VOCAB_FORWARD_DICT)
	# dev_start_index = len(training_dict)
	# test_start_index = dev_start_index + len(development_dict)
	# development_vectors = encode(development_dict, training_vocab_forward_dict, start_index=dev_start_index)
	# test_vectors = encode(test_dict, training_vocab_forward_dict, start_index=test_start_index)


	# development_labels = get_data_labels(development_dict, start_index=dev_start_index)
	
	# DEVELOPMENT_VECTORS = "development_vectors.pickle"
	# DEVELOPMENT_LABELS = "development_labels.pickle"
	# TEST_VECTORS = "test_vectors.pickle"
	
	# save_file(DEVELOPMENT_VECTORS, development_vectors)
	# save_file(DEVELOPMENT_LABELS, development_labels)
	# save_file(TEST_VECTORS, test_vectors)

	# TRAINING_DICTIONARY = "training_dictionary.pickle"
	# training_dict = open_file(TRAINING_DICTIONARY)
	# forward_dict, backward_dict, counts = build_indices(training_dict,0)
	# training_vectors = encode(training_dict, forward_dict)
	# training_vectors, num_changes = introduce_unknown_words(training_vectors, counts)
	# print(num_changes)
	# TRAINING_VECTORS = "training_vectors.pickle"
	# save_file(TRAINING_VECTORS, training_vectors)

	# TRAINING_VOCAB_BACKWARD_DICT = "training_vocab_backward_dictionary.pickle"
	# save_file(TRAINING_VOCAB_FORWARD_DICT, forward_dict)
	# save_file(TRAINING_VOCAB_BACKWARD_DICT, backward_dict)
	
	# training_vocab_backward_dict = open_file(TRAINING_VOCAB_BACKWARD_DICT)

	# training_dict = open_file(TRAINING_DICTIONARY)
	# labels = get_data_labels(training_dict)
	# print(len(labels))
	# LABELS = "labels.pickle"
	# save_file(LABELS, labels)

	# training_vectors = encode(training_dict,training_vocab_forward_dict)
	# save_file(TRAINING_VECTORS, training_vectors)

	TRAINING_VOCAB_FORWARD_DICT = "training_vocab_forward_dictionary.pickle"
	LABELS = "labels.pickle"
	TRAINING_VECTORS = "training_vectors.pickle"

	training_vocab_forward_dict = open_file(TRAINING_VOCAB_FORWARD_DICT)
	training_vectors = open_file(TRAINING_VECTORS)
	labels = open_file(LABELS)
	# print(labels)
	m = model(vocab_size = len(training_vocab_forward_dict), embed_dim=50, context_dim=50, out_dim=2)
	# PATH = 'trained_model_3epoch.pt'
	# m.load_state_dict(torch.load(PATH))
	optimizer = optim.Adam(m.parameters())
	OPTIMIZER_PATH = 'optimizer_state_epoch3.pt'
	torch.save(optimizer.state_dict(), OPTIMIZER_PATH)

	minibatch_size = 508
	num_minibatches = len(training_vectors) // minibatch_size
	num_epochs = 3
	
	for epoch in (range(num_epochs)):
		# Training
		print("Training")
		# Put the model in training mode
		m.train()
		start_train = time.time()

		for group in tqdm(range(num_minibatches)):
			predictions = None
			gold_outputs = None
			loss = 0
			optimizer.zero_grad()
			for i in range(group * minibatch_size, (group + 1) * minibatch_size):
				question_tensor = torch.tensor(training_vectors[i][0])
				context_tensor = torch.tensor(training_vectors[i][1])
				true_label = torch.tensor(labels[i])
				# print(context_tensor.shape)
				# print(question_tensor.shape)
				prediction_vec, pred = m(context_tensor, question_tensor)
				if predictions is None:
					predictions = [prediction_vec]
					gold_outputs = [true_label] 
				else:
					predictions.append(prediction_vec)
					gold_outputs.append(true_label)
			loss = m.compute_loss(torch.stack(predictions), torch.stack(gold_outputs).squeeze())
			loss.backward()
			optimizer.step()
		MODEL_PATH = 'trained_model_'+str(epoch)+'epoch3.pt'
		OPTIMIZER_PATH = 'optimizer_state_'+str(epoch)+'epoch3.pt'
		torch.save(m.state_dict(), MODEL_PATH)
		torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
		print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))
	
	
	# TRAINING_VOCAB_FORWARD_DICT = "training_vocab_forward_dictionary.pickle"
	# DEVELOPMENT_VECTORS = "development_vectors.pickle"
	# DEVELOPMENT_LABELS = "development_labels.pickle"
	# TEST_VECTORS = "test_vectors.pickle"
	# TEST_DICTIONARY = "test_dictionary.pickle"
	
	# test_dictionary = open_file(TEST_DICTIONARY)
	# training_vocab_forward_dict = open_file(TRAINING_VOCAB_FORWARD_DICT)
	# dev_vectors = open_file(DEVELOPMENT_VECTORS)
	# dev_labels = open_file(DEVELOPMENT_LABELS)
	# test_vectors = open_file(TEST_VECTORS)

	# test_start_index = len(training_vectors) + len(dev_vectors)
	# pred_ids = get_pred_ids(test_dictionary,start_index=test_start_index)

	# m = model(vocab_size = len(training_vocab_forward_dict), embed_dim=50, context_dim=50, out_dim=2)
	# PATH = 'trained_model_1epoch.pt'
	# m.load_state_dict(torch.load(PATH))

	# Evaluation
	# print("Evaluation")
	# Put the model in evaluation mode
	# m.eval()
	# start_eval = time.time()
	# predictions = []
	# correct = 0 # number of tokens predicted correctly
	# n = 0
	# num_vectors = len(test_vectors)
	# for test_vector in test_vectors:
	# 	question = torch.tensor(test_vector[0])
	# 	context = torch.tensor(test_vector[1])
	# 	_, prediction = m(context,question)
	# 	predictions.append(prediction)
	# 	n = n + 1
	# 	print(n)
	# print("Evaluation time: {}".format(time.time() - start_eval))

	# predictions = np.array(predictions).reshape((num_vectors,1))
	# pred_ids = np.array(pred_ids).reshape((num_vectors,1))
	# data = np.concatenate((pred_ids,predictions),axis=1)

	# df = pd.DataFrame(data, columns = ['Id', 'Category'])

	# TEST_PREDICTIONS = "predictions.csv"
	# df.to_csv(TEST_PREDICTIONS, index=False)
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_utils import get_word_ids

from config import *
cf = Config()

# Converts a dataset to a numpy format so it can be loaded into the DataLoader.
def convert_format(sentences, word_to_ix):
	data_x = []
	data_y = []
	for sent in sentences:
		npz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)			
		npz[:len(sent)] = np.asarray([word_to_ix[w] for w in sent], )
		data_x.append(npz)
		#data_y.append(word_to_ix[sent[1]])
		next_words = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)	
		for i, word in enumerate(sent[:-1]):
			next_words[i] = word_to_ix[sent[i + 1]]
		data_y.append(next_words)		
	return np.stack(data_x), np.asarray(data_y)

# Found at https://github.com/guillaumegenthial/sequence_tagging
def get_trimmed_glove_vectors(filename):
	"""
	Args:
		filename: path to the npz file
	Returns:
		matrix of embeddings (np array)
	"""
	with np.load(filename) as data:
		return data["embeddings"]


class MyDataset(Dataset):
	def __init__(self, x, y):
		super(MyDataset, self).__init__()
		self.x = x
		self.y = y

	def __getitem__(self, ids):
		return self.x[ids], self.y[ids]

	def __len__(self):
		return self.x.shape[0]

def load_data():
	with open("asset/sentences.pkl", 'r') as f:
		sentences = pkl.load(f)
	with open("asset/word_to_ix.pkl", 'r') as f:
		word_to_ix = pkl.load(f)
	with open("asset/ix_to_word.pkl", 'r') as f:
		ix_to_word = pkl.load(f)
	data_x, data_y = convert_format(sentences, word_to_ix)
	myDataset = MyDataset(data_x, data_y)
	data_iterator = DataLoader(myDataset, batch_size=cf.BATCH_SIZE, pin_memory=True)
	glove_embeddings = get_trimmed_glove_vectors('asset/glove_trimmed.npz')
	logger.info("Loaded %d batches." % len(data_iterator)) 
	logger.info("(%d x %d = ~%d sentences total)" % (len(data_iterator), cf.BATCH_SIZE, len(data_iterator) * cf.BATCH_SIZE))
	logger.info("Loaded %d glove embeddings." % len(glove_embeddings)) 
	return data_iterator, glove_embeddings, word_to_ix, ix_to_word

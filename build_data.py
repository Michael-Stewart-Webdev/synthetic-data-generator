from config import * 
cf = Config()



import pickle as pkl
import numpy as np
from data_utils import get_word_ids, clean_sentences
from get_nltk_sents import get_nltk_sents


# Found at https://github.com/guillaumegenthial/sequence_tagging
def get_glove_vocab(filename):
	logger.info("Building vocab...")
	vocab = set()
	with open(filename) as f:
		for line in f:
			word = line.strip().split(' ')[0]
			vocab.add(word)
	logger.info("Done. Glove vocab contains {} tokens".format(len(vocab)))
	return vocab


# Found at https://github.com/guillaumegenthial/sequence_tagging
def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
	"""
	Saves glove vectors in numpy array

	Args:
		vocab: dictionary vocab[word] = index
		glove_filename: a path to a glove file
		trimmed_filename: a path where to store a matrix in npy
		dim: (int) dimension of embeddings
	"""
	logger.info("Generating trimmed glove vectors...")
	embeddings = np.zeros([len(vocab), dim])

	embeddings[0] = np.zeros(cf.EMBEDDING_DIM) # add zero embeddings for padding

	with open(glove_filename) as f:
		for line in f:
			line = line.strip().split(' ')
			word = line[0]
			embedding = [float(x) for x in line[1:]]
			if word in vocab:
				word_idx = vocab[word]
				embeddings[word_idx] = np.asarray(embedding)	

	np.savez_compressed(trimmed_filename, embeddings=embeddings)
	logger.info("Saved %d glove vectors to %s." % (len(embeddings), trimmed_filename))

def main():
	logger.info("Loading NLTK corpora...")
	sentences = clean_sentences(get_nltk_sents())#[:30000]
	with open("asset/sentences.pkl", 'w') as f:
		pkl.dump(sentences, f)

	glove_vocab = get_glove_vocab('glove/glove.6B.%dd.txt' % cf.EMBEDDING_DIM)
	logger.info("%d sentences after cleaning." % len(sentences))
	word_to_ix, ix_to_word = get_word_ids(sentences)
	with open("asset/word_to_ix.pkl", 'w') as f:
		pkl.dump(word_to_ix, f)
	with open("asset/ix_to_word.pkl", 'w') as f:
		pkl.dump(ix_to_word, f)
	export_trimmed_glove_vectors(word_to_ix, 'glove/glove.6B.%dd.txt' % cf.EMBEDDING_DIM,
									 'asset/glove_trimmed.npz', cf.EMBEDDING_DIM)

if __name__ == "__main__":
	main()
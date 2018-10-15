import torch
from data_utils import *
from config import *
from load_data import load_data
from model import *

def evaluate_model(model, ix_to_word):
	with torch.no_grad():
		# while True:
		# 	sent = raw_input("Enter a sentence: > ").split()
		# 	inputs = prepare_sequence(sent, word_to_ix)
		# 	prediction = model.evaluate(inputs)

		# 	prediction = ix_to_word[prediction]

		# 	print prediction
		logger.info("Generated sentences: ")
		print " " * 6 + "=" * 60
		for x in range(10):
			sent = model.generate_sentence(ix_to_word)			
			print " " * 6 + sent
		print " " * 6 + "=" * 60
		print ""

def main():
	data_iterator, glove_embeddings, word_to_ix, ix_to_word = load_data()

	model = LSTMTagger(cf.EMBEDDING_DIM, cf.HIDDEN_DIM, len(word_to_ix), cf.BATCH_SIZE, cf.MAX_SENT_LENGTH, glove_embeddings)
	model.cuda()
	model.load_state_dict(torch.load('asset/model_trained'))

	evaluate_model(model, ix_to_word)

if __name__ == '__main__':
	main()
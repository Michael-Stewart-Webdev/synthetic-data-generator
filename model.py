import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, max_sent_length, glove_embeddings=None):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.max_sent_length = max_sent_length
		#print glove_embeddings.size

		#num_embeddings, embedding_dim = glove_embeddings.size

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		
		#self.word_embeddings.load_state_dict({ 'weight': glove_embeddings })
		#self.word_embeddings.weight.requires_grad = False

		self.word_embeddings.weight.data.copy_(torch.from_numpy(glove_embeddings))

		# Ensure the word embeddings layer is not trained
		#self.word_embeddings.weight.requires_grad = False


		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)

		#print "lstm"
		#print self.lstm

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim, vocab_size)

		#print "hidden2tag"
		#print self.hidden2tag
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(1, self.batch_size, self.hidden_dim, device=device),
				torch.zeros(1, self.batch_size, self.hidden_dim, device=device))

	def predict_next_word(self, sentence, top_k = 3):

		# Create a 'pretend' batch. The first sentence in the batch is the actual sentence, while the rest are just 0s.
		# There is probably a better way to do this.
		npz = np.zeros((self.batch_size, self.max_sent_length), dtype=int)			
		npz[0, :len(sentence)] = np.asarray(sentence)
		batch = torch.tensor(npz).to(device)

		batch_lengths = []
		for x in batch:
			batch_lengths.append(len(x))

		# print batch, batch_lengths, batch.size()

		prediction =  self.forward(batch, batch_lengths)

		# #print prediction
		# print "..."
		# for p in prediction:
		# 	print p

		# print "---"

		#prediction = prediction[0].narrow(1, 1, vocab_size - 1)

		# We are only interested in the last timestep, i.e. the timestep corresponding to the last word in the sentence.
		v, wi = prediction[0][len(sentence)-1].topk(top_k)
		
		#print wi
		wi = random.sample(wi, 1)[0]	# Pick a random one from the top k
		# wi is now the top k indexes 
		
		#v, wi = prediction[0][len(sentence)-1].max(0)

		#print "W", wi
		#print v, wi
		#wi = wi.cpu().numpy().item(0)
		return wi

	def generate_sentence(self, ix_to_word):
		sentence = []
		sentence.append(random.randint(0, self.vocab_size - 1))
		sentence = torch.tensor(sentence, dtype=torch.long, device=device)

		for x in range(self.max_sent_length):
			next_word = self.predict_next_word(sentence)
			sentence = torch.cat((sentence, next_word.view(1)), 0)

		readable_sentence = []
		for word in sentence:
			readable_sentence.append(ix_to_word[word.cpu().numpy().item(0)])
		
		return " ".join(readable_sentence)




	def forward(self, batch, batch_lengths):
		# embeds = self.word_embeddings(sentence)
		# lstm_out, self.hidden = self.lstm(
		# 	embeds.view(len(sentence), 1, -1), self.hidden)
		# tag_space = self.hidden2tag(lstm_out[-1])
		# return tag_space

		self.hidden = self.init_hidden()
		
		batch_size, seq_len = batch.size()

		# print ">>", batch_size, "<>", seq_len

		# 1. Embed the input
		batch = self.word_embeddings(batch)

		# 2. Pack the sequence
		batch = torch.nn.utils.rnn.pack_padded_sequence(batch, batch_lengths, batch_first=True)

		# 3. Run through lstm
		batch, self.hidden = self.lstm(batch, self.hidden)

		# Undo packing
		batch, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first = True)

		# print batch
		# print batch.shape

		batch = batch.contiguous()
		batch = batch.view(-1, batch.shape[2])

		# print batch
		# print batch.shape

		# run through actual linear layer
		#tag_space = self.hidden2tag(batch[-1])
		batch = self.hidden2tag(batch)

		# print batch
		# print batch.shape
		# print "><"

		#print tag_space.squeeze(), tag_space.shape

		batch = F.log_softmax(batch, dim=1)

		#print batch.shape

		batch = batch.view(batch_size, seq_len, self.vocab_size)

		#print batch

		#batch = batch[:, [timestep], :] # We are only interested in the last timestep, hence the [-1]

		#print "!!!"
		#print batch
		# print batch
		# print batch.shape
		# print ">>"

		return batch


# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
def modified_loss(Y_hat, Y, X_lengths, word_to_ix):
	# TRICK 3 ********************************
	# before we calculate the negative log likelihood, we need to mask out the activations
	# this means we don't want to take into account padded items in the output vector
	# simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
	# and calculate the loss on that.

	# flatten all the labels
	Y = Y.view(-1).to(device)

	# flatten all predictions
	Y_hat = Y_hat.view(-1, len(word_to_ix))

	# create a mask by filtering out all tokens that ARE NOT the padding token
	tag_pad_token = word_to_ix['<PAD>']
	mask = (Y > tag_pad_token).float()

	# count how many tokens we have
	nb_tokens = int(torch.sum(mask).item())

	# pick the values for the label and zero out the rest with the mask
	Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

	# compute cross entropy loss which ignores all <PAD> tokens
	ce_loss = -torch.sum(Y_hat) / nb_tokens

	return ce_loss
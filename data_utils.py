import torch
from config import *
cf = Config()

def get_word_ids(sentences):
	word_to_ix = { "<PAD>": 0 }
	for sent in sentences:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
	#word_to_ix[SOS_TOKEN] = len(word_to_ix)
	#word_to_ix[EOS_TOKEN] = len(word_to_ix)
	ix_to_word = {v: k for k, v in word_to_ix.iteritems()}
	return word_to_ix, ix_to_word

def clean_sentences(sentences):
	clean_sents = []
	for sent in sentences:
		clean_sent = [w.lower() for w in sent if w.isalpha()]
		if len(clean_sent) >= cf.MIN_SENT_LENGTH and len(clean_sent) <= cf.MAX_SENT_LENGTH:
			clean_sents.append(clean_sent)
	return clean_sents

# Segment data into training/test (probably not necessary for generating sentences)
def segment_data(sentences):
	l = len(sentences)
	logger.info("Loaded %d sentences." % l)
	return sentences[:int(len(sentences) * 0.9)], sentences[int(len(sentences) * 0.9):]

# Convert a tokenized sentence into a tensor of word indexes.
def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long, device=device)

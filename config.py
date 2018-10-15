import logging as logger
import sys, torch
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoggingFormatter(logger.Formatter):
    def format(self, record):
        #compute s according to record.levelno
        #for example, by setting self._fmt
        #according to the levelno, then calling
        #the superclass to do the actual formatting
        if record.levelno == 10:
        	return Fore.CYAN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL,  record.msg, )
        else:
        	return Fore.GREEN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg) 
        #return s

hdlr = logger.StreamHandler(sys.stdout)
hdlr.setFormatter(LoggingFormatter())
logger.root.addHandler(hdlr)
logger.root.setLevel(logger.DEBUG)
#logger.setFormatter(LoggingFormatter())

class Config():

	def __init__(self):

		self.LOAD_PRETRAINED_MODEL = False
		self.TRIAL_MODE = True

		self.EMBEDDING_DIM 		= 300	# The number of dimensions to use for embeddings. Can be either 100 or 300.
		self.HIDDEN_DIM 		= 200	# The number of dimensions of the hidden layer.
		self.BATCH_SIZE 		= 32	# The batch size (larger uses more memory but is faster)

		self.MIN_SENT_LENGTH 	= 3		# The minimum length of a sentence. Sentences smaller than this will not be trained on.
		self.MAX_SENT_LENGTH 	= 5	# The maximum length of a sentence. Sentences larger than this will not be trained on.
		self.MAX_EPOCHS 		= 300	# The maximum number of epochs to run.
		self.EARLY_STOP			= True  # Whether to stop when no progress has been made for the last 10 epochs. (i.e. loss has not improved)


		# self.SOS_TOKEN = "<SOS>"
		# self.EOS_TOKEN = "<EOS>"



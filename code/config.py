MAX_LEN = 160
TRAIN_BATCH_SIZE = 4 
VALID_BATCH_SIZE = 4
TRAIN_EPOCHS = 10
GRADIENT_ACCUMULATION_CONSTANT = 3
MODEL_SAVE_PATH = 'output/best_model'
MODEL_NAME = 'bert-base-cased'
BASE_DIR = '/home/shinigami/Desktop/code_dir/interviews/nlp_test/Entity-based-sentiment/'
TRAIN_FILE_PATH = BASE_DIR + 'data/train.xlsx'
TEST_FILE_PATH = BASE_DIR + 'data/test.xlsx'
PROCESSED_TRAIN_FILE_PATH = BASE_DIR + 'data/processed_train.csv'
PROCESSED_TEST_FILE_PATH = BASE_DIR + 'data/processed_test.csv'
FROZEN_BERT_EARLY_STOPPING = 3
FROZEN_BERT_EPOCHS = 20
BERT_EPOCHS = 5
BERT_EARLY_STOPPING = 1
TEST_FILE_SAVE_PATH='output/output_test_file.csv'
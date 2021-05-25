from transformers import AlbertTokenizer, AlbertModel


def tokenizer(albert_tokenizer_dir):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    tokenizer.save_pretrained(albert_tokenizer_dir)


def model(albert_model_dir):
    model = AlbertModel.from_pretrained('albert-base-v2')
    # model.save_pretrained('/data/pretrained_models/albert_model/')
    model.save_pretrained(albert_model_dir)

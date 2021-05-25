**This code is for paper`TWAG: A topic-guided wikipedia abstract generator`**

**Python version**: Python3.6

**Package Requirements**: pytorch  transformers spacy  tensorboardX  

# Get Start

First, clone our code from github:

```bash
git clone xxxx.git
```

Enter TWAG's root directory. All command then should be executed here.

```bash
cd TWAG
```



# Classify-Stage Preprocess

To prepare data for Classify-Stage Training:

```bash
python -m src.prepare --data_dir DATA_DIR \
--classifier_dir CLASSIFIER_DIR\
--generator_dir GENERATOR_DIR \
--tokenizer_dir TOKENIZER_DIR
```

- DATA_DIR is the directory where you put the download dataset. Dataset can be downloaded from [google drive](https://drive.google.com/file/d/1gw_j_3rF38boFaTurCrHR4MMqJAc7-CU/view?usp=sharing). The size of this dataset is 6.9 GB.
- CLASSIFIER_DIR is the directory where results of classifier go. If CLASSIFIER_DIR doesn't exist, we will create it.
- GENERATOR_DIR is the directory where results of generator go. If GENERATOR_DIR doesn't exist, we will create it.
- TOKENIZER_DIR is the directory of tokenizer files.  If TOKENIZER_DIR doesn't exist, we will create it. Then we will download tokenizer files from huggingface and store them in TOKENIZER_DIR.

Classify-Stage Preprocess usually takes **4** hours.

# Classify-Stage Training

To train with the default setting as in the paper:

```bash
python -m src.classify.train  --category CATEGORY \
--classifier-dir CLASSIFIER_DIR\
--albert-model-dir ALBERT_MODEL_DIR \
--topic-num TOPIC_NUM 
```

- CATEGORY is the category of data's domain, it can be 'animal' or 'film' or 'company'
- CLASSIFIER_DIR is the directory where results of classifier go.  It should be same with last stage's CLASSIFIER_DIR.
- ALBERT_MODEL_DIR is the directory of albert-model files.  If ALBERT_MODEL_DIR doesn't exist, we will create it. Then we will download albert-model files from huggingface and store them in ALBERT_MODEL_DIR.

- TOPIC_NUM is the number of topics for a domain. TOPIC_NUM is decided by the `src/sample/select.py` which has been performed in Classify-Stage Preprocess. 
- For animal and company, the default TOPIC_NUM is 5. 
  - For film, the default TOPIC_NUM is 6. 

It takes **8** hours to finish Classify-Stage Training on animal or company.

It takes **24** hours to finish Classify-Stage Training on film due to the large scale of the film dataset.

# Generation-Stage Preprocess

After training a classify model, we can use it on the data preparation for generation stage.

To prepare data for Generation-Stage Training:

```bash
python -m src.final_prepare --category CATEGORY \
--classifier_dir CLASSIFIER_DIR \
--generator_dir GENERATOR_DIR \
--tokenizer_dir TOKENIZER_DIR \
--albert-model-dir ALBERT_MODEL_DIR \
--glove_path GLOVE_PATH 
```

- CATEGORY is the category of data's domain, it can be 'animal' or 'film' or 'company'
- CLASSIFIER_DIR is the directory where results of classifier go.  It should be same with last stage's CLASSIFIER_DIR.
- GENERATOR_DIR is the directory where results of generator go.  It should be same with last stage's GENERATOR_DIR.
- TOKENIZER_DIR is the directory of tokenizer files.  It should be same with last stage's TOKENIZER_DIR.
- ALBERT_MODEL_DIR is the directory of albert-model files.  It should be same with last stage's ALBERT_MODEL_DIR.
- GLOVE_PATH is the path where your GloVe vector file  `glove.840B.300d.py36.pkl` is. The glove file can be downloaded from [google drive](https://drive.google.com/file/d/1V7zNHxi92gBWGvNhvvZ7JaCmUbKMuPF5/view?usp=sharing). The size of the glove file is 5.1 GB.

It takes **13** hours to finish Generation-Stage Preprocess on animal.

It takes **20** hours to finish Generation-Stage Preprocess on company.

It takes **18** hours to finish Generation-Stage Preprocess on film.

# Generation-Stage Training

To train with the default setting as in the paper:

```bash
python -m src.c_generate_soft.train  --category CATEGORY \
--classifier_dir CLASSIFIER_DIR \
--generator_dir GENERATOR_DIR \
--is_coverage \
--topic-num TOPIC_NUM
```

- CATEGORY is the category of data's domain, it can be 'animal' or 'film' or 'company'
- CLASSIFIER_DIR is the directory where results of classifier go.  It should be same with last stage's CLASSIFIER_DIR.
- GENERATOR_DIR is the directory where results of generator go.  It should be same with last stage's GENERATOR_DIR.
- The flag `--is_coverage`  means we will use the coverage loss in `TopicDecodeModel`
- TOPIC_NUM is the number of topics for a domain. TOPIC_NUM is decided by the `src/sample/select.py` which has been performed in Classify-Stage Preprocess. 
  - For animal and company, the default TOPIC_NUM is 5. 
  - For film, the default TOPIC_NUM is 6. 

It takes **3** hours to finish every epoch of Generation-Stage Training on animal.

It takes **4.5** hours to finish every epoch of Generation-Stage Training on company.

It takes **3.5** hours to finish every epoch of Generation-Stage Training on film.

# Evaluation

To evaluate the trained model on test set:

```bash
python -m src.c_generate_soft.validate --category CATEGORY \
--classifier_dir CLASSIFIER_DIR \
--generator_dir GENERATOR_DIR \
--test \
--ckpt CKPT_PATH
```

- CATEGORY is the category of data's domain, it can be 'animal' or 'film' or 'company'
- CLASSIFIER_DIR is the directory where results of classifier go.  It should be same with last stage's CLASSIFIER_DIR.
- GENERATOR_DIR is the directory where results of generator go.  It should be same with last stage's GENERATOR_DIR.
- The flag `--test`  means we will evaluate on test set.
- CKPT_PATH is the absolute path of  the model's checkpoint 
  - for example: `/data/generator/film/trained_generate_models/6_topics/20_titles/model_epoch8_val0.404.pkl`

The checkpoint from CKPT_PATH will be evaluated by ROUGE using `pyrouge`, the score and abstracts will be saved at a result directory which is named after the checkpoint, for example: `/data/generator/film/trained_generate_models/6_topics/20_titles/test_id_model_epoch8_val0.404`

Under this result directory, there will be:

- scores.txt
  - scores.txt records the  ROUGE score result.
- sum/
  - Under sum directory, there are all summaries generated during evaluation.
- ref/
  - Under ref directory, there are all reference golden summaries. 

It takes **7** hours to finish Evaluation on animal.

It takes **18** hours to finish Evaluation on company.

It takes **6** hours to finish Evaluation on film.



# End2End generation

To generate abstract in end2end way:

```bash
python -m src.end2end --category CATEGORY \
--tmp_dir TMP_DIR \
--data_dir DATA_DIR \
--topic_file_path TOPIC_FILE_DIR \
--id_file_path ID_FILE_DIR \
--classify_ckpt_path  \
--generator_ckpt_path /data1/tsq/TWAG/data/generator/animal/trained_generate_models/5_topics/20_titles/model_epoch6_val0.446.pt \
--tokenizer_dir TOKENIZER_DIR \
--albert-model-dir ALBERT_MODEL_DIR \
--glove_path GLOVE_PATH 
```

- CATEGORY is the category of data's domain, it can be 'animal' or 'film' or 'company'
- TMP_DIR is where the temporary files go.
- DATA_DIR is where the src and tgt file exists.
- TOPIC_FILE_DIR is where  `TopicList.txt`　file exists.
- ID_FILE_DIR is where  `id.pkl`　file exists, it is under the `GENERATOR_DIR/CATEGORY` in training stage.
-  CLASSIFY_CKPT_PATH is where the checkpoint of classify stage exists. 
  - For example, `/data/classifier/classifier_animal/checkpoints/3/bert_classifier.pkl`
- GENERATOR_CKPT_PATH is the absolute path of  the generator model's checkpoint 
  - for example: `/data/generator/animal/trained_generate_models/5_topics/20_titles/model_epoch6_val0.446.pt`
- TOKENIZER_DIR is the directory of tokenizer files.  It should be same with last stage's TOKENIZER_DIR.
- ALBERT_MODEL_DIR is the directory of albert-model files.  It should be same with last stage's ALBERT_MODEL_DIR.
- GLOVE_PATH is the path where your glove file  `glove.840B.300d.py36.pkl` is. The glove file can be downloaded from [google drive](https://drive.google.com/file/d/1V7zNHxi92gBWGvNhvvZ7JaCmUbKMuPF5/view?usp=sharing). The size of the glove file is 5.1 GB.

By using end2end generation, the abstract of the src file in DATA_DIR will be generated and saved at a result directory which is named after the GENERATOR_CKPT_PATH, for example: `/data/generator/animal/trained_generate_models/5_topics/20_titles/test_id_model_epoch6_val0.446`


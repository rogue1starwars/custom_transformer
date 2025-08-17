This repo is a pytorch implementation of the Transformer paper. [Read the original paper](https://arxiv.org/abs/1706.03762)

It contains the implementation of the model itself along with the code and scripts to train and evaluate the model for a Japanese-to-English translation task.

The dataset used for training is [JESC (Japanese-English Subtitle Corpus)](https://nlp.stanford.edu/projects/jesc/) from Stanford University.

## Usage

### Installing required libraries

```bash
pip install -r requirements.txt
```

Before training, setup your python environment by installing the required libraries listed in requirements.txt.

### Configuring the model

To configure the model, you need to modify the `config.yaml` file. This file contains all the hyperparameters and settings for the model, including the number of layers, hidden size, and dropout rates. Make sure to adjust these settings according to your specific requirements and hardware capabilities.

### Training

To train the model, run the following command:

```bash
python3 train.py --config config.yaml
```

This will start the training process using the settings specified in the `config.yaml` file. Make sure to monitor the training progress and adjust the hyperparameters as needed.

For each epoch, the model weights will be stored in the `pretrained_models` directory. The final weights will also be saved in this directory after training is complete.

### Inferencing

To test the trained model, run the following command:

```bash
python3 translate.py --config config.yaml
```

This will load the trained model and perform inference on the provided Japanese text.

Example output:

```
Extracting config file...
Using device: cpu
Loading tokenizers...
Extracting data...
Raw data already exists at ./data/raw.tar.gz, skipping download.
Raw data already extracted at ./data_raw/raw/raw, skipping extraction.
Loading vocabulary...
Finished.
Vocabulary sizes:
142816
140774
Loading model...
Starting translation, input 'exit' when exiting.
Input Japanese Sentence:おはようございます。今日は良い天気ですね。
Japanese: おはようございます。今日は良い天気ですね。
English: good morning . today is a good weather .
Input Japanese Sentence:あなたの好きな食べ物はなんですか?
Japanese: あなたの好きな食べ物はなんですか?
English: what is your favorite food ?
Input Japanese Sentence:exit
Ending translation...
```

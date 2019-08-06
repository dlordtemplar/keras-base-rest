# %%
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from loading_preprocessing_TC import *
import pickle
import json
import os
import nltk

MODEL_DIR = 'out/data/semeval/models'
DATASET_PATH = 'resources/datasets/semeval/train/'
DATA_PATH = 'out/data/semeval/'
MODEL_PATH = 'out/data/semeval/models/'
NEURON_COUNT_PATH = 'out/data/semeval/neuron_count.json'

MAX_LENGTH = 200
model = None
tokenizer = None
embeddings = None
vocabulary_encoded = None
vocabulary_inv = None
qa_pairs = None
answer_texts = None
graph = None

NEURON_MAX = 128


def load_data():
    """Load SemEval 2017 files from .xml and convert them into pandas dataframes.
    Args:

    Returns:
        train (pandas dataframe): QA-pairs in a format question - correct answers (ids) - pool (ids; incorrect answers).
        If there are multiple correct answers to a single question, they are split into multiple QA - pairs.
        answer_texts_train (pandas dataframe): answer texts and their ids.
    """
    files = [DATASET_PATH + 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
             DATASET_PATH + 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml']
    train_xml = read_xml(files)
    train, answer_texts_train = xml2dataframe_Labels(train_xml, 'train')
    answer_texts_train.set_index('answer_id', drop=False, inplace=True)
    return train, answer_texts_train


def load_model(new_model_filename):
    """Load a pretrained model from PyTorch / Keras checkpoint.
    Args:
        new_model_filename (string): the name of the model used when saving its weights and architecture to
        either a binary (PyTorch) or a .h5 and a .json (Keras)

    Returns:
        error (string): The error message displayed to a user. If empty, counts as no error.
    """
    global model, model_filename
    print("Loading model:", new_model_filename)
    try:
        json_file = open(MODEL_PATH + new_model_filename + '.json',
                         'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        global graph
        graph = tf.get_default_graph()
        # load weights into new model
        model.load_weights(MODEL_PATH + new_model_filename + ".h5")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_filename = new_model_filename
        return model
    except Exception as e:
        print(e)
        error = "<div class=\"alert alert-warning\"> Sorry, there is something wrong with the model: <br> " + str(
            e) + "</div>"
        return error


def load_environment():
    """Load documents index for search engine, pre-trained embeddings, vocabulary, parameters and the model."""
    global model, tokenizer, embeddings, vocabulary_encoded, vocabulary_inv, qa_pairs, answer_texts, graph
    with open(DATA_PATH + 'tokenizer.p', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(DATA_PATH + 'embedding_matrix.p', 'rb') as handle:
        embeddings = pickle.load(handle)
    vocabulary_encoded = tokenizer.word_index
    vocabulary_inv = {v: k for k, v in vocabulary_encoded.items()}
    model = load_model('model_visualization_siamesedeeplstm')
    qa_pairs, answer_texts = load_data()

    return model


def prepare_data(texts):
    """Tokenize texts and pad resulting sequences of words using Keras functions."""
    global tokenizer, embeddings
    tokens = tokenizer.texts_to_sequences(texts)
    padded_tokens = pad_sequences(tokens, maxlen=MAX_LENGTH, value=embeddings.shape[0] - 1)
    return tokens, padded_tokens


def visualize_model_deep(model, question_lstm=True):
    """Retrieve weights of the second shared LSTM to visualize neuron activations."""
    recurrent_layer = model.get_layer('SharedLSTM2')
    output_layer = model.layers[-1]

    inputs = []
    inputs.extend(model.inputs)

    outputs = []
    outputs.extend(model.outputs)
    if question_lstm:
        outputs.append(recurrent_layer.get_output_at(1))
    else:
        outputs.append(recurrent_layer.get_output_at(0))

    global graph
    with graph.as_default():
        all_function = K.function(inputs, outputs)
        output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function


def highlight_neuron(rnn_values, texts, tokens, scale, neuron):
    """Generate HTML code where each word is highlighted according to a given neuron activity on it."""
    tag_string = "<span data-toggle=\"tooltip\" title=\"SCORE\"><span style = \"background-color: rgba(COLOR, OPACITY);\">WORD</span></span>"
    old_texts = texts
    texts = []
    for idx in range(0, len(old_texts)):
        current_neuron_values = rnn_values[idx, :, neuron]
        current_neuron_values = current_neuron_values[-len(tokens[idx]):]
        words = [vocabulary_inv[x] for x in tokens[idx]]
        current_strings = []
        if scale:
            scaled = [
                ((x - min(current_neuron_values)) * (2) / (
                        max(current_neuron_values) - min(current_neuron_values))) + (
                    -1)
                for x in current_neuron_values]
        else:
            scaled = current_neuron_values
        for score, word, scaled_score in zip(current_neuron_values, words, scaled):
            if score > 0:
                color = '195, 85, 58'
            else:
                color = '63, 127, 147'
            current_string = tag_string.replace('SCORE', str(score)).replace('WORD', word).replace('OPACITY', str(
                abs(scaled_score))).replace('COLOR', color)
            current_strings.append(current_string)
        texts.append(' '.join(current_strings))
    return texts


model = load_environment()
print("Finished loading.")
# %%
# Generate pos_tagged_questions


# nltk.download('averaged_perceptron_tagger')
POS_TAGGED_QUESTIONS_PATH = 'out/data/semeval/pos_tagged_questions.json'

print(len(qa_pairs))
if not os.path.isfile(POS_TAGGED_QUESTIONS_PATH):
    print('File not found! Creating new version...')
    tagged_dict = {}
    for i in range(len(qa_pairs)):
        row = qa_pairs.iloc[i]
        question = row['question']

        tokens = nltk.word_tokenize(question)
        tagged = nltk.pos_tag(tokens)
        tagged_dict[i] = tagged
    print('Writing to file...')
    with open(POS_TAGGED_QUESTIONS_PATH, 'w') as file:
        json.dump(tagged_dict, file)
        print('Finished!')
else:
    print('Loading existing file.')
    with open(POS_TAGGED_QUESTIONS_PATH, 'r') as file:
        tagged_dict = json.load(file)

# %%

indices = [0, 1]
neuron = 0

# Start actual visualization
all_highlighted_wrong_answers = []
all_wrong_answers = []

min_ca = 1
min_wa = 1
max_ca = -1
max_wa = -1

activated_words = []
activated_words_values = []
antiactivated_words = []
antiactivated_words_values = []

activation_per_word_data = {}
asked_questions = {}

all_function_deep, output_function_deep = visualize_model_deep(model, False)
nlp = spacy.load('en')

"""
neuron_counts

[
    {
        num: 0,
        # tokens = [('Yes', 'UH'), ('Try', 'VB'), ...]
        token_counts: {
            UH: 1,
            VB: 1,
            ...
        }
    },
    ...
]
"""
neuron_counts = []
for neuron_num in range(NEURON_MAX):
    print('neuron', neuron_num)
    neuron_count = {
        'neuron': neuron_num,
        # 'tokens': [],
        'token_counts': {}
    }
    for i in range(1790):  # indices:
        print('Generating activations for QA pair', i)
        row = qa_pairs.iloc[i]
        correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
        # correct_answers_pos = []
        # for sent in correct_answers:
        #     # tokens = nltk.word_tokenize(sent)
        #     tokens = nlp(sent)
        #     nlp_tokens = [str(token.text) for token in tokens]
        #     correct_answers_pos.append(nltk.pos_tag(nlp_tokens))
        wrong_answers = answer_texts.loc[row['pool']]['answer'].values
        question = row['question']
        asked_questions[i] = question
        q_tokens, q_padded_tokens = prepare_data([question])
        ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
        wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)
        if len(correct_answers) > 0:
            scores_ca, rnn_values_ca = all_function_deep([q_padded_tokens * len(correct_answers), ca_padded_tokens])
            # neuron_num = rnn_values_ca.shape[-1]
            all_values_ca = rnn_values_ca[:, :, neuron_num:neuron_num + 1]
            # if np.min(all_values_ca) < min_ca:
            #     min_ca = np.min(all_values_ca)
            # if np.max(all_values_ca) > max_ca:
            #     max_ca = np.max(all_values_ca)

            for idx in range(0, len(correct_answers)):
                current_neuron_values = rnn_values_ca[idx, :, neuron_num]
                current_neuron_values = current_neuron_values[-len(ca_tokens[idx]):]
                # print(correct_answers[idx]) # Yes. It is right behind Kahrama in the National area.
                # print(ca_tokens[idx]) # [163, 11, 7, 139, 651, 12356, 6, 1, 1083, 363]
                # print(current_neuron_values) # [5.8287954e-01 2.0389782e-02 3.1098869e-04 3.2598805e-04 1.8663764e-04 1.3387189e-03 1.0951190e-04 6.6894456e-04 5.4823589e-03 5.3216936e-03]

                tokens = keras.preprocessing.text.text_to_word_sequence(correct_answers[idx], lower=True)
                tagged = nltk.pos_tag(tokens)

                if len(tagged) > 0:

                    val_max = max(current_neuron_values)
                    val_min = min(current_neuron_values)
                    index_min = np.argmin(current_neuron_values)
                    index_max = np.argmax(current_neuron_values)
                    # print('max', val_max)
                    # print('min', val_min)
                    index_chosen = -1
                    if abs(val_min) > val_max:
                        index_chosen = index_min
                    else:
                        index_chosen = index_max
                    # print(index_chosen)
                    # print(keras.preprocessing.text.text_to_word_sequence(correct_answers[idx], lower=True))

                    # print(tagged)
                    token_to_add = tagged[index_chosen]
                    # print(token_to_add)
                    # neuron_count['tokens'].append(token_to_add)
                    if token_to_add[1] in neuron_count['token_counts']:
                        neuron_count['token_counts'][token_to_add[1]] = neuron_count['token_counts'][token_to_add[1]] + 1
                    else:
                        neuron_count['token_counts'][token_to_add[1]] = 1
                else:
                    print('No tagged tokens!')

        else:
            pass

        if len(wrong_answers) > 0:
            scores_wa, rnn_values_wa = all_function_deep([q_padded_tokens * len(wrong_answers), wa_padded_tokens])

            for idx in range(0, len(wrong_answers)):
                current_neuron_values = rnn_values_wa[idx, :, neuron_num]
                current_neuron_values = current_neuron_values[-len(wa_tokens[idx]):]

                tokens = keras.preprocessing.text.text_to_word_sequence(wrong_answers[idx], lower=True)
                tagged = nltk.pos_tag(tokens)

                if len(tagged) > 0:
                    val_max = max(current_neuron_values)
                    val_min = min(current_neuron_values)
                    index_min = np.argmin(current_neuron_values)
                    index_max = np.argmax(current_neuron_values)
                    # print('max', val_max)
                    # print('min', val_min)
                    index_chosen = -1

                    if abs(val_min) > val_max:
                        index_chosen = index_min
                    elif abs(val_min) < val_max:
                        index_chosen = index_max
                    else:
                        pass
                    # print(index_chosen)

                    # print('wa')
                    # print(wrong_answers)
                    # print(wrong_answers[idx])

                    # print(tagged)
                    # print(index_chosen)
                    token_to_add = tagged[index_chosen]
                    # print(token_to_add)
                    # neuron_count['tokens'].append(token_to_add)
                    if token_to_add[1] in neuron_count['token_counts']:
                        neuron_count['token_counts'][token_to_add[1]] = neuron_count['token_counts'][token_to_add[1]] + 1
                    else:
                        neuron_count['token_counts'][token_to_add[1]] = 1
                else:
                    print('No tagged tokens!')

        else:
            pass

        # print(neuron_count)
    neuron_counts.append(neuron_count)

with open(NEURON_COUNT_PATH, 'w') as file:
    json.dump(neuron_counts, file)

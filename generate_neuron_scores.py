# %%
from sklearn.manifold import TSNE

import notebook_util

print(notebook_util.list_available_gpus())
notebook_util.setup_one_gpu()
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from loading_preprocessing_TC import *
import pickle
import json
import os

start_time = time.time()

MODEL_DIR = 'out/data/semeval/models'
DATASET_PATH = 'resources/datasets/semeval/train/'
DATA_PATH = 'out/data/semeval/'
MODEL_PATH = 'out/data/semeval/models/'
FLATTENED_PATH = 'out/data/semeval/flattened.json'
TSNE_NEURONS_PARTIAL_PATH = 'out/data/semeval/tsne_neurons_p'
TSNE_NEURONS_POINTS_PARTIAL_PATH = 'out/data/semeval/tsne_neurons_points_p'

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
udpipe_URL = 'http://lindat.mff.cuni.cz/services/udpipe/api/process'
munderline_URL = 'http://iread.dfki.de/munderline/de_ud'


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


def get_neuron_attention_per_token(rnn_values, texts, tokens, neuron):
    result = []
    for idx in range(len(texts)):
        current_neuron_values = rnn_values[idx, :, neuron]
        current_neuron_values = current_neuron_values[-len(tokens[idx]):]
        words = [vocabulary_inv[x] for x in tokens[idx]]

        current_strings = []
        for score, word in zip(current_neuron_values, words):
            current_string = (word, str(score))
            current_strings.append(current_string)
        result.append(current_strings)
    return result


indices = [0, 1]

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

neuron_activations = []
if os.path.exists(FLATTENED_PATH):
    print('Loading existing flattened.json ...')
    with open(FLATTENED_PATH, 'r') as file:
        neuron_activations = json.load(file)
    print('Done.')
else:
    print('Generating new file...')
    model = load_environment()
    print("Finished loading model.")
    all_function_deep, output_function_deep = visualize_model_deep(model, False)
    nlp = spacy.load('en')
    for neuron_num in range(NEURON_MAX):
        # for neuron_num in range(2):
        # print('neuron', neuron_num)
        neuron_vector = []
        for i in range(0, len(qa_pairs)):  # indices:
            # for i in range(0, 2):  # indices:
            print('Generating activations for QA pair', i)

            row = qa_pairs.iloc[i]
            correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
            wrong_answers = answer_texts.loc[row['pool']]['answer'].values
            question = row['question']
            asked_questions[i] = question

            q_tokens, q_padded_tokens = prepare_data([question])
            ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
            wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)

            if len(correct_answers) > 0:
                scores_ca, rnn_values_ca = all_function_deep(
                    [q_padded_tokens * len(correct_answers), ca_padded_tokens])

                tuples = get_neuron_attention_per_token(rnn_values_ca, correct_answers,
                                                        ca_tokens, neuron_num)

                for idx in range(len(correct_answers)):
                    for tuple in tuples[idx]:
                        neuron_vector.append(tuple[1])
            else:
                pass

            if len(wrong_answers) > 0:
                scores_wa, rnn_values_wa = all_function_deep(
                    [q_padded_tokens * len(wrong_answers), wa_padded_tokens])

                tuples = get_neuron_attention_per_token(rnn_values_wa, wrong_answers, wa_tokens,
                                                        neuron_num)
                for idx in range(len(wrong_answers)):
                    for tuple in tuples[idx]:
                        neuron_vector.append(tuple[1])
            else:
                pass
        neuron_activations.append(neuron_vector)
        # current_time = time.time()
        # print('Elapsed time:', current_time - start_time, 'seconds')
    end_time = time.time()
    print('Total time:', end_time - start_time, 'seconds')
    with open(FLATTENED_PATH, 'w') as file:
        json.dump(neuron_activations, file)

perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 45]
for perplexity in perplexities:
    full_TSNE_neurons_path = TSNE_NEURONS_PARTIAL_PATH + str(perplexity) + '.json'
    if os.path.exists(full_TSNE_neurons_path):
        print('Loading existing', full_TSNE_neurons_path, '...')
        with open(full_TSNE_neurons_path, 'r') as file:
            new_values = json.load(file)
        print('Done.')
    else:
        print('Generating new file...')
        tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23,
                          metric="cosine")
        # list of x y values in format (x, y)
        new_values = tsne_model.fit_transform(neuron_activations)

        with open(full_TSNE_neurons_path, 'w') as file:
            json.dump(new_values.tolist(), file)

    full_TSNE_neurons_path = TSNE_NEURONS_POINTS_PARTIAL_PATH + str(perplexity) + '.json'

    if os.path.exists(full_TSNE_neurons_path):
        print(full_TSNE_neurons_path, 'already exists. Continuing...')
    else:
        trace_question_x = []
        trace_question_y = []
        trace_question_text = []

        for i in range(len(new_values)):
            trace_question_x.append(new_values[i][0])
            trace_question_y.append(new_values[i][1])
            trace_question_text.append('Neuron ' + str(i))

        marker_magenta = {
            'size': 20,
            'color': 'rgb(255,0,255)',
            # star
            'symbol': 'triangle-up'
        }
        trace_question = {
            'name': 'Neuron',
            'x': trace_question_x,
            'y': trace_question_y,
            'type': 'scatter',
            'text': trace_question_text,
            'mode': 'markers+text',
            'hoverinfo': 'text',
            'marker': marker_magenta
        }

        plotly_tsne = [trace_question]
        plotly_tsne_as_json = pd.Series(plotly_tsne).to_json(orient='values')

        with open(full_TSNE_neurons_path, 'w') as file:
            json.dump(plotly_tsne_as_json, file)
            print('Wrote', full_TSNE_neurons_path)

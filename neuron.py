import random
import json
import pickle

import notebook_util

notebook_util.setup_one_gpu()

import tensorflow as tf
from flask import (
    Blueprint, request, jsonify
)
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE

from loading_preprocessing_TC import *

bp = Blueprint('neuron', __name__)
MODEL_DIR = 'out/data/semeval/models'
DATASET_PATH = 'resources/datasets/semeval/train/'
DATA_PATH = 'out/data/semeval/'
MODEL_PATH = 'out/data/semeval/models/'

# model
model = None
tokenizer = None
embeddings = None
vocabulary_encoded = None
vocabulary_inv = None
qa_pairs = None
answer_texts = None
graph = None

# deep learning settings
MAX_LENGTH = 200

# defaults values for the visualization pages
DEFAULT_NUM_TEXTS = 5

# Pair
DEFAULT_PERPLEXITY = 5


@bp.before_app_first_request
def setup():
    global model
    if not model:
        model = load_environment()


# global model, tokenizer, embeddings, vocabulary_encoded, vocabulary_inv, qa_pairs, answer_texts, graph


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


@bp.route('/neuron', strict_slashes=False, methods=['GET', 'POST'])
def display_neuron():
    global answer_texts, qa_pairs, vocabulary_inv, model

    print(request.data)
    data = json.loads(request.data)
    indices = data['indices']
    neuron = data['neuron']

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

    # plotly
    pl_ca_heatmaps_indexed = {}
    pl_wa_heatmaps_indexed = {}
    indexed_correct_answers = {}
    indexed_highlighted_correct_answers = {}
    indexed_wrong_answers = {}
    indexed_highlighted_wrong_answers = {}

    for i in indices:
        print('Generating activations for QA pair', i)
        row = qa_pairs.iloc[i]
        correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
        wrong_answers = answer_texts.loc[row['pool']]['answer'].values
        question = row['question']
        asked_questions[i] = question
        q_tokens, q_padded_tokens = prepare_data([question])
        ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
        wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)
        all_function_deep, output_function_deep = visualize_model_deep(model, False)
        if len(correct_answers) > 0:
            scores_ca, rnn_values_ca = all_function_deep([q_padded_tokens * len(correct_answers), ca_padded_tokens])
            neuron_num = rnn_values_ca.shape[-1]
            all_values_ca = rnn_values_ca[:, :, neuron:neuron + 1]
            if np.min(all_values_ca) < min_ca:
                min_ca = np.min(all_values_ca)
            if np.max(all_values_ca) > max_ca:
                max_ca = np.max(all_values_ca)
            highlighted_correct_answers = highlight_neuron(rnn_values_ca, correct_answers, ca_tokens,
                                                           False,  # Scale placeholder
                                                           neuron)

            if i not in indexed_highlighted_correct_answers:
                indexed_highlighted_correct_answers[i] = [highlighted_correct_answers]
            else:
                indexed_highlighted_correct_answers[i].append(highlighted_correct_answers)

            current_ca = [[vocabulary_inv[x] for x in ca_tokens[idx]] for idx in range(len(ca_tokens))]
            if i not in indexed_correct_answers:
                indexed_correct_answers[i] = current_ca
            else:
                indexed_correct_answers[i].append(current_ca)

            activation_per_word_data['ca_firings' + str(i)] = rnn_values_ca[:, :, neuron].flatten()
            activation_per_word_data['ca_text' + str(i)] = [
                vocabulary_inv[token] if token in vocabulary_inv.keys() else '<pad>' for x in ca_padded_tokens for
                token
                in x]
        else:
            if i not in indexed_highlighted_correct_answers:
                indexed_highlighted_correct_answers[i] = []
            else:
                indexed_highlighted_correct_answers[i].append([])

            if i not in indexed_correct_answers:
                indexed_correct_answers[i] = []
            else:
                indexed_correct_answers[i].append([])

            activation_per_word_data['ca_text' + str(i)] = []
            activation_per_word_data['ca_firings' + str(i)] = []

        if len(wrong_answers) > 0:
            scores_wa, rnn_values_wa = all_function_deep([q_padded_tokens * len(wrong_answers), wa_padded_tokens])
            neuron_num = rnn_values_wa.shape[-1]
            all_values_wa = rnn_values_wa[:, :, neuron:neuron + 1]
            if np.min(all_values_wa) < min_wa:
                min_wa = np.min(all_values_wa)
            if np.max(all_values_wa) > max_wa:
                max_wa = np.max(all_values_wa)
            highlighted_wrong_answers = highlight_neuron(rnn_values_wa, wrong_answers, wa_tokens, False,
                                                         # Scale placeholder
                                                         neuron)
            all_highlighted_wrong_answers.append(highlighted_wrong_answers)

            if i not in indexed_highlighted_wrong_answers:
                indexed_highlighted_wrong_answers[i] = [highlighted_wrong_answers]
            else:
                indexed_highlighted_wrong_answers[i].append(highlighted_wrong_answers)

            current_wa = [[vocabulary_inv[x] for x in wa_tokens[idx]] for idx in range(len(wa_tokens))]
            if i not in indexed_wrong_answers:
                indexed_wrong_answers[i] = current_wa
            else:
                indexed_wrong_answers[i].append(current_wa)

            activation_per_word_data['wa_firings' + str(i)] = rnn_values_wa[:, :, neuron].flatten()
            activation_per_word_data['wa_text' + str(i)] = [
                vocabulary_inv[token] if token in vocabulary_inv.keys() else '<pad>' for x in wa_padded_tokens for
                token
                in x]
        else:
            all_highlighted_wrong_answers.append([])

            if i not in indexed_highlighted_wrong_answers:
                indexed_highlighted_wrong_answers[i] = []
            else:
                indexed_highlighted_wrong_answers[i].append([])

            all_wrong_answers.append([])

            if i not in indexed_wrong_answers:
                indexed_wrong_answers[i] = []
            else:
                indexed_wrong_answers[i].append([])

            activation_per_word_data['wa_text' + str(i)] = []
            activation_per_word_data['wa_firings' + str(i)] = []

        # Point generation for correct answers
        if len(correct_answers) > 0:
            for idx in range(0, len(ca_tokens)):
                words = [vocabulary_inv[x] for x in ca_tokens[idx]]
                heatmap_points = {'z': rnn_values_ca[idx, -len(ca_tokens[idx]):, neuron:neuron + 1].tolist(),
                                  'y': words,
                                  'type': 'heatmap'}
                if i in pl_ca_heatmaps_indexed:
                    pl_ca_heatmaps_indexed[i].append(heatmap_points)
                else:
                    pl_ca_heatmaps_indexed[i] = [heatmap_points]

        # Same as above, but for wrong answers
        if len(wrong_answers) > 0:
            for idx in range(0, len(wa_tokens)):
                words = [vocabulary_inv[x] for x in wa_tokens[idx]]
                heatmap_points = {'z': rnn_values_wa[idx, -len(wa_tokens[idx]):, neuron:neuron + 1].tolist(),
                                  'y': words,
                                  'type': 'heatmap'}
                if i in pl_wa_heatmaps_indexed:
                    pl_wa_heatmaps_indexed[i].append(heatmap_points)
                else:
                    pl_wa_heatmaps_indexed[i] = [heatmap_points]

    all_firings = [x for i in indices for x in activation_per_word_data['wa_firings' + str(i)]] + [x for
                                                                                                   i in indices
                                                                                                   for x
                                                                                                   in
                                                                                                   activation_per_word_data[
                                                                                                       'ca_firings' + str(
                                                                                                           i)]]
    all_tokens = [x for i in indices for x in activation_per_word_data['wa_text' + str(i)]] + [x for i in
                                                                                               indices
                                                                                               for x in
                                                                                               activation_per_word_data[
                                                                                                   'ca_text' + str(
                                                                                                       i)]]
    all_firings = np.array(all_firings)
    all_tokens = np.array(all_tokens)
    p_high = np.percentile([x for i, x in enumerate(all_firings) if all_tokens[i] != '<pad>'], 90)
    p_low = np.percentile([x for i, x in enumerate(all_firings) if all_tokens[i] != '<pad>'], 10)

    for ind, x in enumerate(all_firings):
        if x >= p_high:
            activated_words.append(all_tokens[ind])
            activated_words_values.append(x)
        elif x <= p_low:
            antiactivated_words.append(all_tokens[ind])
            antiactivated_words_values.append(x)

    seen = set()
    activated_words = [x for x in activated_words if not (x in seen or seen.add(x))]
    seen = set()
    antiactivated_words = [x for x in antiactivated_words if not (x in seen or seen.add(x))]

    return jsonify({'max_qa_pairs': len(qa_pairs),
                    'activated_words': activated_words,
                    'antiactivated_words': antiactivated_words,
                    'asked_questions': json.dumps(asked_questions),
                    # plotly
                    'pl_ca_heatmap_points': pl_ca_heatmaps_indexed,
                    'pl_wa_heatmap_points': pl_wa_heatmaps_indexed,
                    'indexed_correct_answers': indexed_correct_answers,
                    'indexed_highlighted_correct_answers': indexed_highlighted_correct_answers,
                    'indexed_wrong_answers': indexed_wrong_answers,
                    'indexed_highlighted_wrong_answers': indexed_highlighted_wrong_answers
                    })


def tsne_plot(model, labels, correct_answers, wrong_answers, question, perplexity=40):
    """Creates a TSNE model and plots it"""

    tokens = []
    for word in model.keys():
        tokens.append(model[word])

    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23,
                      metric="cosine")
    # list of x y values in format (x, y)
    new_values = tsne_model.fit_transform(tokens)

    trace_question_x = []
    trace_ca_x = []
    trace_wa_x = []
    trace_question_y = []
    trace_ca_y = []
    trace_wa_y = []
    trace_question_text = []
    trace_ca_text = []
    trace_wa_text = []
    trace_question_hovertext = []
    trace_ca_hovertext = []
    trace_wa_hovertext = []

    ca_index = 0
    wa_index = 0
    for label_index in range(len(labels)):
        if labels[label_index] == 'q':
            trace_question_x.append(new_values[label_index][0])
            trace_question_y.append(new_values[label_index][1])
            trace_question_text.append('Q')
            trace_question_hovertext.append(question if len(question) < 61 else question[:60] + '...')
        elif labels[label_index] == 'ca':
            trace_ca_x.append(new_values[label_index][0])
            trace_ca_y.append(new_values[label_index][1])
            trace_ca_text.append('CA' + str(len(trace_ca_x)))
            trace_ca_hovertext.append(
                correct_answers[ca_index] if len(correct_answers[ca_index]) < 61 else correct_answers[ca_index][
                                                                                      :60] + '...')
            ca_index += 1
        elif labels[label_index] == 'wa':
            trace_wa_x.append(new_values[label_index][0])
            trace_wa_y.append(new_values[label_index][1])
            trace_wa_text.append('WA' + str(len(trace_wa_x)))
            trace_wa_hovertext.append(
                wrong_answers[wa_index] if len(wrong_answers[wa_index]) < 61 else wrong_answers[wa_index][:60] + '...')
            wa_index += 1

    marker_blue = {
        'size': 20,
        'color': 'rgb(0, 0, 255)',
        # star
        'symbol': 17
    }
    marker_green = {
        'size': 20,
        'color': 'rgb(0, 204, 0)',
        # circle
        'symbol': 0
    }
    marker_red = {
        'size': 20,
        'color': 'rgb(255, 0, 0)',
        # x
        'symbol': 4
    }
    trace_question = {
        'name': 'Question',
        'x': trace_question_x,
        'y': trace_question_y,
        'type': 'scatter',
        'mode': 'markers+text',
        'hoverinfo': 'text',
        'hovertext': trace_question_hovertext,
        'text': trace_question_text,
        'textposition': 'top right',
        'marker': marker_blue
    }
    trace_ca = {
        'name': 'Correct answer',
        'x': trace_ca_x,
        'y': trace_ca_y,
        'type': 'scatter',
        'mode': 'markers+text',
        'hoverinfo': 'text',
        'hovertext': trace_ca_hovertext,
        'text': trace_ca_text,
        'textposition': 'top right',
        'marker': marker_green
    }
    trace_wa = {
        'name': 'Wrong answer',
        'x': trace_wa_x,
        'y': trace_wa_y,
        'type': 'scatter',
        'mode': 'markers+text',
        'hoverinfo': 'text',
        'hovertext': trace_wa_hovertext,
        'text': trace_wa_text,
        'textposition': 'top right',
        'marker': marker_red
    }
    plotly_tsne = [trace_question, trace_ca, trace_wa]

    plotly_tsne_as_json = pd.Series(plotly_tsne).to_json(orient='values')

    return plotly_tsne_as_json


@bp.route('/pair', strict_slashes=False, methods=['GET', 'POST'])
def pair():
    global answer_texts, qa_pairs, vocabulary_inv, model

    data = json.loads(request.data)
    print(data)

    pair_num = data['pair_num']
    perplexity = data['perplexity']
    scale = data['scale']
    neuron_display_ca = data['ca_neuron']
    if neuron_display_ca > -1:
        neuron_display_ca = int(neuron_display_ca)
    neuron_display_wa = data['wa_neuron']
    if neuron_display_wa > -1:
        neuron_display_wa = int(neuron_display_wa)

    if pair_num >= len(qa_pairs):
        return 'Index out of bounds.'

    row = qa_pairs.iloc[pair_num]
    correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
    wrong_answers = answer_texts.loc[row['pool']]['answer'].values
    question = row['question']
    q_tokens, q_padded_tokens = prepare_data([question])
    ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
    wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)
    all_function_deep, output_function_deep = visualize_model_deep(model, False)
    if len(correct_answers) > 0:
        scores_ca, rnn_values_ca = all_function_deep([q_padded_tokens * len(correct_answers), ca_padded_tokens])
        neuron_num = rnn_values_ca.shape[-1]
    else:
        scores_ca = []
        rnn_values_ca = []
    if len(wrong_answers) > 0:
        scores_wa, rnn_values_wa = all_function_deep([q_padded_tokens * len(wrong_answers), wa_padded_tokens])
        neuron_num = rnn_values_wa.shape[-1]
    else:
        scores_wa = []
        rnn_values_wa = []

    # generate TSNE
    labels = ['q'] + ['ca'] * len(correct_answers) + ['wa'] * len(wrong_answers)
    model_dict_wa = {}
    model_dict_ca = {}
    if len(correct_answers) > 0:
        model_dict_ca = {i + 1: np.max(rnn_values_ca[i, :, :], axis=1) for i in range(len(correct_answers))}
    if len(wrong_answers) > 0:
        model_dict_wa = {i + 1: np.max(rnn_values_wa[i - len(correct_answers), :, :], axis=1) for i in
                         range(len(correct_answers), len(wrong_answers) + len(correct_answers))}
    model_dict = {**model_dict_ca, **model_dict_wa}
    all_function_deep_q, output_function_deep_q = visualize_model_deep(model, True)
    _, rnn_values = all_function_deep_q([q_padded_tokens, [ca_padded_tokens[0]]])
    question_vector = rnn_values[0]
    model_dict[0] = np.max(question_vector, axis=1)
    plotly_tsne = tsne_plot(model_dict, labels, correct_answers, wrong_answers, question, perplexity)

    # plotly
    pl_ca_heatmaps = []
    pl_wa_heatmaps = []
    # generate heatmaps
    # plotly
    if len(correct_answers) > 0:
        for idx in range(0, len(ca_tokens)):
            words = [vocabulary_inv[x] for x in ca_tokens[idx]]
            heatmap_points = {'z': rnn_values_ca[idx, -len(ca_tokens[idx]):, :].tolist(),
                              'y': words,
                              'type': 'heatmap'}
            pl_ca_heatmaps.append(heatmap_points)
    # Same as above, but for wrong answers
    if len(wrong_answers) > 0:
        for idx in range(0, len(wa_tokens)):
            words = [vocabulary_inv[x] for x in wa_tokens[idx]]
            heatmap_points = {'z': rnn_values_wa[idx, -len(wa_tokens[idx]):, :].tolist(),
                              'y': words,
                              'type': 'heatmap'}
            pl_wa_heatmaps.append(heatmap_points)

    # generate text highlighting based on neuron activity
    highlighted_correct_answers = correct_answers.tolist()
    highlighted_wrong_answers = wrong_answers.tolist()

    if neuron_display_ca > -1:
        highlighted_correct_answers = highlight_neuron(rnn_values_ca, correct_answers, ca_tokens, scale,
                                                       neuron_display_ca)
    if neuron_display_wa > -1:
        highlighted_wrong_answers = highlight_neuron(rnn_values_wa, wrong_answers, wa_tokens, scale, neuron_display_wa)

    # Convert ndarrays to lists
    if len(scores_ca) > 0:
        scores_ca = scores_ca.tolist()
    if len(scores_wa) > 0:
        scores_wa = scores_wa.tolist()

    return jsonify({'question': question,
                    'highlighted_wrong_answers': highlighted_wrong_answers,
                    'highlighted_correct_answers': highlighted_correct_answers,
                    'wrong_answers': wrong_answers.tolist(),
                    'correct_answers': correct_answers.tolist(),
                    'pair_num': pair_num,
                    'neuron_num': neuron_num,
                    'neuron_display_ca': neuron_display_ca,
                    'neuron_display_wa': neuron_display_wa,
                    'scale': scale,
                    'texts_len': len(qa_pairs),
                    'scores_ca': scores_ca,
                    'scores_wa': scores_wa,
                    # plotly
                    'plotly_tsne': plotly_tsne,
                    'pl_ca_heatmaps': pl_ca_heatmaps,
                    'pl_wa_heatmaps': pl_wa_heatmaps
                    })


@bp.route('/live/load', strict_slashes=False, methods=['GET', 'POST'])
def live_load():
    global qa_pairs

    data = json.loads(request.data)

    pair_num = data['pair_num']

    row = qa_pairs.iloc[pair_num]
    correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
    wrong_answers = answer_texts.loc[row['pool']]['answer'].values
    question = row['question']

    return jsonify({'question': question,
                    'wrong_answers': wrong_answers.tolist(),
                    'correct_answers': correct_answers.tolist(),
                    'pair_num': pair_num,
                    })


@bp.route('/live/random', strict_slashes=False, methods=['GET', 'POST'])
def live_random():
    global qa_pairs

    data = json.loads(request.data)

    pair_num = random.randint(0, len(qa_pairs) - 1)

    row = qa_pairs.iloc[pair_num]
    correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
    wrong_answers = answer_texts.loc[row['pool']]['answer'].values
    question = row['question']

    return jsonify({'question': question,
                    'wrong_answers': wrong_answers.tolist(),
                    'correct_answers': correct_answers.tolist(),
                    'pair_num': pair_num,
                    })


@bp.route('/live', strict_slashes=False, methods=['GET', 'POST'])
def live():
    global answer_texts, qa_pairs, vocabulary_inv, model

    data = json.loads(request.data)

    perplexity = data['perplexity']
    scale = data['scale']
    neuron_display = data['neuron']
    if neuron_display > -1:
        neuron_display = int(neuron_display)

    correct_answers = []
    for ca in data['correct_answers'].split('\n'):
        if ca.strip() != '':
            correct_answers.append(ca)
    wrong_answers = []
    for wa in data['wrong_answers'].split('\n'):
        if wa.strip() != '':
            wrong_answers.append(wa)
    question = data['question']
    q_tokens, q_padded_tokens = prepare_data([question])
    ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
    wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)
    all_function_deep, output_function_deep = visualize_model_deep(model, False)
    if len(correct_answers) > 0:
        scores_ca, rnn_values_ca = all_function_deep([q_padded_tokens * len(correct_answers), ca_padded_tokens])
        neuron_num = rnn_values_ca.shape[-1]
    else:
        scores_ca = []
        rnn_values_ca = []
    if len(wrong_answers) > 0:
        scores_wa, rnn_values_wa = all_function_deep([q_padded_tokens * len(wrong_answers), wa_padded_tokens])
        neuron_num = rnn_values_wa.shape[-1]
    else:
        scores_wa = []
        rnn_values_wa = []

    # generate TSNE
    labels = ['q'] + ['ca'] * len(correct_answers) + ['wa'] * len(wrong_answers)
    model_dict_wa = {}
    model_dict_ca = {}
    if len(correct_answers) > 0:
        model_dict_ca = {i + 1: np.max(rnn_values_ca[i, :, :], axis=1) for i in range(len(correct_answers))}
    if len(wrong_answers) > 0:
        model_dict_wa = {i + 1: np.max(rnn_values_wa[i - len(correct_answers), :, :], axis=1) for i in
                         range(len(correct_answers), len(wrong_answers) + len(correct_answers))}
    model_dict = {**model_dict_ca, **model_dict_wa}
    all_function_deep_q, output_function_deep_q = visualize_model_deep(model, True)
    _, rnn_values = all_function_deep_q([q_padded_tokens, [ca_padded_tokens[0]]])
    question_vector = rnn_values[0]
    model_dict[0] = np.max(question_vector, axis=1)
    plotly_tsne = tsne_plot(model_dict, labels, correct_answers, wrong_answers, question, perplexity)

    # plotly
    pl_ca_heatmaps = []
    pl_wa_heatmaps = []
    # generate heatmaps
    # plotly
    if len(correct_answers) > 0:
        for idx in range(0, len(ca_tokens)):
            words = [vocabulary_inv[x] for x in ca_tokens[idx]]
            heatmap_points = {'z': rnn_values_ca[idx, -len(ca_tokens[idx]):, :].tolist(),
                              'y': words,
                              'type': 'heatmap'}
            pl_ca_heatmaps.append(heatmap_points)
    # Same as above, but for wrong answers
    if len(wrong_answers) > 0:
        for idx in range(0, len(wa_tokens)):
            words = [vocabulary_inv[x] for x in wa_tokens[idx]]
            heatmap_points = {'z': rnn_values_wa[idx, -len(wa_tokens[idx]):, :].tolist(),
                              'y': words,
                              'type': 'heatmap'}
            pl_wa_heatmaps.append(heatmap_points)

    # generate text highlighting based on neuron activity
    highlighted_correct_answers = correct_answers
    highlighted_wrong_answers = wrong_answers

    if neuron_display > -1:
        highlighted_correct_answers = highlight_neuron(rnn_values_ca, correct_answers, ca_tokens, scale,
                                                       neuron_display)
        highlighted_wrong_answers = highlight_neuron(rnn_values_wa, wrong_answers, wa_tokens, scale, neuron_display)

    # Convert ndarrays to lists
    if len(scores_ca) > 0:
        scores_ca = scores_ca.tolist()
    if len(scores_wa) > 0:
        scores_wa = scores_wa.tolist()

    return jsonify({'question': question,
                    'highlighted_wrong_answers': highlighted_wrong_answers,
                    'highlighted_correct_answers': highlighted_correct_answers,
                    'wrong_answers': wrong_answers,
                    'correct_answers': correct_answers,
                    'perplexity': perplexity,
                    'neuron_display': neuron_display,
                    'scale': scale,
                    'texts_len': len(qa_pairs),
                    'scores_ca': scores_ca,
                    'scores_wa': scores_wa,
                    # plotly
                    'plotly_tsne': plotly_tsne,
                    'pl_ca_heatmaps': pl_ca_heatmaps,
                    'pl_wa_heatmaps': pl_wa_heatmaps
                    })


@bp.route('/questions', strict_slashes=False, methods=['GET', 'POST'])
def questions():
    with open('out/data/plotly_all_questions_json_string_p20.json', 'r') as file:
        plotly_all_questions = json.load(file)

    return plotly_all_questions


@bp.route('/pos/ptb', strict_slashes=False, methods=['GET', 'POST'])
def pos_ptb():
    with open('out/data/semeval/pos_per_neuron_ptb.json', 'r') as file:
        plotly_pos = json.load(file)
    return plotly_pos


@bp.route('/pos/upos', strict_slashes=False, methods=['GET', 'POST'])
def pos_upos():
    with open('out/data/semeval/pos_per_neuron_upos.json', 'r') as file:
        plotly_pos = json.load(file)
    return plotly_pos


@bp.route('/pos/xpos', strict_slashes=False, methods=['GET', 'POST'])
def pos_xpos():
    with open('out/data/semeval/pos_per_neuron_xpos.json', 'r') as file:
        plotly_pos = json.load(file)
    return plotly_pos


@bp.route('/tsne_neurons/<perplexity>', strict_slashes=False, methods=['GET', 'POST'])
def tsne_neurons(perplexity):
    with open('out/data/semeval/tsne_neurons_points_p' + str(perplexity) + '.json', 'r') as file:
        plotly_tsne_neurons = json.load(file)
    return plotly_tsne_neurons

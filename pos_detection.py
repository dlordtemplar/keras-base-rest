# %%
from requests.adapters import HTTPAdapter

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
import requests
from requests.exceptions import ConnectionError
from urllib3.util.retry import Retry

from spacy.gold import align

# bert_tokens = ["obama", "'", "s", "podcast"]
# spacy_tokens = ["obama", "'s", "podcast"]
# alignment = align(bert_tokens, spacy_tokens)
# cost, a2b, b2a, a2b_multi, b2a_multi = alignment

start_time = time.time()

MODEL_DIR = 'out/data/semeval/models'
DATASET_PATH = 'resources/datasets/semeval/train/'
DATA_PATH = 'out/data/semeval/'
MODEL_PATH = 'out/data/semeval/models/'
NEURON_COUNT_PATH = 'out/data/semeval/neuron_count.json'
POS_PER_NEURON_PATH = 'out/data/semeval/pos_per_neuron.json'
POS_DICT_PATH = 'out/data/semeval/pos_dict.json'

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


def convert_texts_to_POS(texts, pos_dict, type, qa_pair_num):
    texts_POS_raw = []
    texts_POS_tokens = []
    for idx in range(0, len(texts)):
        ud_output = get_ud_POS_data(texts[idx], pos_dict, type, qa_pair_num, idx)
        temp_raw = []
        temp_tokens = []
        for line in ud_output.split('\n'):
            tab_split = line.split('\t')
            if not tab_split[0].startswith('#') and tab_split[0].strip() != '':
                temp_raw.append(tab_split)
                temp_tokens.append(tab_split[1])
        texts_POS_raw.append(temp_raw)
        texts_POS_tokens.append(' '.join(temp_tokens))
    return texts_POS_raw, texts_POS_tokens


def get_ud_POS_data(tokens, pos_dict, type, qa_pair_num, answer_num):
    if str(qa_pair_num) in pos_dict and type in pos_dict[str(qa_pair_num)] and str(answer_num) in \
            pos_dict[str(qa_pair_num)][type]:
        parser_output = pos_dict[str(qa_pair_num)][type][str(answer_num)]
    else:
        data = {
            'data': tokens,
            'model': 'english-gum-ud-2.4-190531',
            'tokenizer': '',
            'tagger': '--tag',
            'parser': ''
        }

        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        s.mount('http://', HTTPAdapter(max_retries=retries))

        response = s.post(udpipe_URL, headers=headers, data=data)
        response.encoding = 'utf-8'
        parser_output = response.json()['result']
        pos_dict[str(qa_pair_num)][type][str(answer_num)] = parser_output
        print('Saving new pos_dict file...')
        with open(POS_DICT_PATH, 'w') as file:
            json.dump(pos_dict, file)
        print('Done.')
    return parser_output


def get_neuron_attention_per_token(rnn_values, texts, tokens, neuron):
    result = []
    all_tokens = []
    for idx in range(0, len(texts)):
        current_neuron_values = rnn_values[idx, :, neuron]
        current_neuron_values = current_neuron_values[-len(tokens[idx]):]
        words = [vocabulary_inv[x] for x in tokens[idx]]

        current_strings = []
        for score, word in zip(current_neuron_values, words):
            current_string = (word, score)
            current_strings.append(current_string)
        result.append(current_strings)
        all_tokens.append(words)
    return result, all_tokens


def convert_from_ud_to_array(raw_ud_input):
    result = []
    for line in raw_ud_input.split('\n'):
        if not line.startswith('#') and line.strip() != '':
            result.append(line.split())
    return result


def align_tokens_and_ud(token_score_tuples, ud_output):
    result = []
    score_tuples_a = []
    ud_out_b = []
    for tuple in token_score_tuples:
        score_tuples_a.append(tuple[0].lower())
    for row in ud_output:
        ud_out_b.append(row[1].lower())
    alignment = align(score_tuples_a, ud_out_b)
    cost, a2b, b2a, a2b_multi, b2a_multi = alignment

    debug_print = False
    for tuple_index in range(len(token_score_tuples)):
        if a2b[tuple_index] != -1:
            result.append((token_score_tuples[tuple_index][0], token_score_tuples[tuple_index][1],
                           ud_output[a2b[tuple_index]][3], ud_output[a2b[tuple_index]][4]))
        else:
            debug_print = True

    # if len(a2b_multi) > 0:
    #     debug_print = True
    #     print('a2b_multi', a2b_multi)
    # if len(b2a_multi) > 0:
    #     debug_print = True
    #     print('b2a_multi', b2a_multi)

    # if debug_print:
    #     print('a', score_tuples_a)
    #     print('a2b', a2b)
    #     print('b', ud_out_b)
    #     print('b2a', b2a)

    return result


# The code below ensures that the POS tagging result and the tokenized results align properly
# def align_tokens_and_ud(token_score_tuples, ud_output):
#     result = []
#     temp = convert_from_ud_to_array(ud_output)
#     pos_extra_index = 0
#     for index in range(len(temp)):
#         if len(token_score_tuples) - 1 >= index and len(temp) + pos_extra_index - 1 >= index:
#             result.append((token_score_tuples[index][0], token_score_tuples[index][1], temp[index + pos_extra_index][3],
#                            temp[index + pos_extra_index][4]))
#             if token_score_tuples[index][0] != temp[index + pos_extra_index][1]:
#                 if len(token_score_tuples) - 1 >= index + 1:
#                     while token_score_tuples[index + 1][0] != temp[index + 1 + pos_extra_index][1]:
#                         pos_extra_index += 1
#     return result


model = load_environment()
print("Finished loading.")

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

all_function_deep, output_function_deep = visualize_model_deep(model, False)
nlp = spacy.load('en')

# %%


"""
neuron_counts

[
    {
        num: 0,
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

if os.path.exists(NEURON_COUNT_PATH):
    print('Loading existing neuron_count.json ...')
    with open(NEURON_COUNT_PATH, 'r') as file:
        neuron_counts = json.load(file)
    print('Done.')
else:
    print('Generating new file...')
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'
    }

    # Every neuron will react to the same tokens/POS tags, so only call to server once per input instead of 1790x
    if os.path.exists(POS_DICT_PATH):
        print('Loading existing pos_dict.json ...')
        with open(POS_DICT_PATH, 'r') as file:
            pos_dict = json.load(file)
        print('Done.')
    else:
        pos_dict = {}

    for neuron_num in range(NEURON_MAX):
    # for neuron_num in range(1):
        print('neuron', neuron_num)
        neuron_count = {
            'neuron': neuron_num,
            'token_counts': {}
        }
        for i in range(0, len(qa_pairs)):  # indices:
            # for i in range(0, 2):  # indices:
            print('Generating activations for QA pair', i)
            if str(i) not in pos_dict:
                pos_dict[str(i)] = {
                    'q': {},
                    'ca': {},
                    'wa': {}
                }

            row = qa_pairs.iloc[i]
            correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
            wrong_answers = answer_texts.loc[row['pool']]['answer'].values
            question = row['question']
            asked_questions[i] = question

            # Make new prepare_data method with POS tokens instead of raw "texts" input
            question_POS, question_POS_tokens = convert_texts_to_POS([[question]], pos_dict, 'q', i)
            correct_answers_POS, correct_answers_POS_tokens = convert_texts_to_POS(correct_answers, pos_dict, 'ca', i)
            wrong_answers_POS, wrong_answers_POS_tokens = convert_texts_to_POS(wrong_answers, pos_dict, 'wa', i)

            # q_tokens, q_padded_tokens = prepare_data([question])
            # ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
            # wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)
            q_tokens, q_padded_tokens = prepare_data([question_POS_tokens])
            ca_tokens, ca_padded_tokens = prepare_data(correct_answers_POS_tokens)
            wa_tokens, wa_padded_tokens = prepare_data(wrong_answers_POS_tokens)

            if len(correct_answers) > 0:
                scores_ca, rnn_values_ca = all_function_deep(
                    [q_padded_tokens * len(correct_answers_POS_tokens), ca_padded_tokens])

                tuples, all_tokens = get_neuron_attention_per_token(rnn_values_ca, correct_answers_POS_tokens,
                                                                    ca_tokens, neuron_num)

                for idx in range(len(all_tokens)):
                    # current_pos_scores = align_tokens_and_ud(tuples[idx], parser_output)
                    current_pos_scores = align_tokens_and_ud(tuples[idx], correct_answers_POS[idx])
                    # [('most', 0.73064023, 'ADJ', 'JJS'), ('of', 0.031687938, 'ADP', 'IN'), ('the', 0.008439351, 'DET', 'DT'), ('time', 7.566358e-05, 'NOUN', 'NN'), ('hijacking', 0.00023871037, 'VERB', 'VBG'), ('shifts', 0.00029278902, 'VERB', 'VBZ'), ('the', 0.00026579967, 'DET', 'DT'), ('main', 0.026925175, 'ADJ', 'JJ'), ('topic', 0.0046378975, 'NOUN', 'NN'), ('to', 0.0003025322, 'ADP', 'TO'), ('a', 0.00088415114, 'DET', 'DT'), ('different', 0.0012040904, 'ADJ', 'JJ'), ('one', 0.009830474, 'NUM', 'CD'), ('and', 0.00029199503, 'CCONJ', 'CC'), ('then', 0.0035359005, 'ADV', 'RB'), ('to', 0.00042696742, 'ADP', 'TO'), ('another', 0.00050246046, 'DET', 'DT'), ('different', 0.0010206797, 'ADJ', 'JJ'), ('one', 0.0062409323, 'NUM', 'CD'), ('and', 0.00012129463, 'CCONJ', 'CC'), ('so', 0.00026837253, 'ADV', 'RB'), ('on', 0.00025421553, 'ADP', 'IN')]
                    for current_tuple in current_pos_scores:
                        # current_tuple[2] = UPOS, current_tuple[3] = XPOS,
                        if current_tuple[3] in neuron_count['token_counts']:
                            neuron_count['token_counts'][current_tuple[3]] = neuron_count['token_counts'][
                                                                                 current_tuple[3]] + abs(
                                current_tuple[1])
                        else:
                            neuron_count['token_counts'][current_tuple[3]] = abs(current_tuple[1])
            else:
                pass

            if len(wrong_answers) > 0:
                scores_wa, rnn_values_wa = all_function_deep(
                    [q_padded_tokens * len(wrong_answers_POS_tokens), wa_padded_tokens])

                tuples, all_tokens = get_neuron_attention_per_token(rnn_values_wa, wrong_answers_POS_tokens, wa_tokens,
                                                                    neuron_num)
                for idx in range(len(all_tokens)):
                    current_pos_scores = align_tokens_and_ud(tuples[idx], wrong_answers_POS[idx])
                    # [('most', 0.73064023, 'ADJ', 'JJS'), ('of', 0.031687938, 'ADP', 'IN'), ('the', 0.008439351, 'DET', 'DT'), ('time', 7.566358e-05, 'NOUN', 'NN'), ('hijacking', 0.00023871037, 'VERB', 'VBG'), ('shifts', 0.00029278902, 'VERB', 'VBZ'), ('the', 0.00026579967, 'DET', 'DT'), ('main', 0.026925175, 'ADJ', 'JJ'), ('topic', 0.0046378975, 'NOUN', 'NN'), ('to', 0.0003025322, 'ADP', 'TO'), ('a', 0.00088415114, 'DET', 'DT'), ('different', 0.0012040904, 'ADJ', 'JJ'), ('one', 0.009830474, 'NUM', 'CD'), ('and', 0.00029199503, 'CCONJ', 'CC'), ('then', 0.0035359005, 'ADV', 'RB'), ('to', 0.00042696742, 'ADP', 'TO'), ('another', 0.00050246046, 'DET', 'DT'), ('different', 0.0010206797, 'ADJ', 'JJ'), ('one', 0.0062409323, 'NUM', 'CD'), ('and', 0.00012129463, 'CCONJ', 'CC'), ('so', 0.00026837253, 'ADV', 'RB'), ('on', 0.00025421553, 'ADP', 'IN')]
                    for current_tuple in current_pos_scores:
                        # current_tuple[2] = UPOS, current_tuple[3] = XPOS,
                        if current_tuple[3] in neuron_count['token_counts']:
                            neuron_count['token_counts'][current_tuple[3]] = neuron_count['token_counts'][
                                                                                 current_tuple[3]] + abs(
                                current_tuple[1])
                        else:
                            neuron_count['token_counts'][current_tuple[3]] = abs(current_tuple[1])
            else:
                pass
        # print(neuron_count)
        neuron_counts.append(neuron_count)
        current_time = time.time()
        print('Elapsed time:', current_time - start_time, 'seconds')

    # %%
    for neuron in neuron_counts:
        total_attention = sum(neuron['token_counts'].values())
        # print(total_attention)
        remainders = {}
        for key in neuron['token_counts']:
            rounded_value = round(neuron['token_counts'][key] / total_attention * 100, 2)
            remainders[key] = rounded_value - np.floor(rounded_value)
            neuron['token_counts'][key] = rounded_value
        # {'JJS': 10.55, 'IN': 7.46, 'DT': 10.7, 'NN': 7.2, 'VBG': 1.67, 'VBZ': 3.76, 'JJ': 4.19, 'TO': 0.09, 'CD': 4.2, 'CC': 2.26, 'RB': 7.34, 'LS': 6.37, 'VBD': 4.64, 'NNS': 1.94, 'VBN': 0.03, 'PRP': 7.16, 'MD': 9.69, 'VB': 3.65, 'VBP': 0.53, 'WP': 0.0, 'PRP$': 0.41, 'UH': 0.31, 'EX': 0.0, 'RP': 1.09, 'WRB': 0.0, 'JJR': 4.8}
        # print(neuron['token_counts'])
        # print(remainders)
        pos_percents_rounded = {}
        for key, value in neuron['token_counts'].items():
            pos_percents_rounded[key] = np.floor(neuron['token_counts'][key])
        remainders_desc = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        # print(sum(neuron['token_counts'].values()))
        # Largest Remainder Method to roughly split values to add up to 100%
        # Also, JSON does not encode float32.
        # https://en.wikipedia.org/wiki/Largest_remainder_method
        # print(remainders_desc)
        # print(pos_percents_rounded)
        # print(100 - sum(pos_percents_rounded.values()))
        to_allocate = 100 - sum(pos_percents_rounded.values())
        for current_tuple in remainders_desc:
            if to_allocate == 0:
                break
            else:
                pos_percents_rounded[current_tuple[0]] += 1
                to_allocate -= 1
        # print(pos_percents_rounded)
        # print(sum(pos_percents_rounded.values()))
        for key in pos_percents_rounded:
            neuron['token_counts'][key] = pos_percents_rounded[key]

    # %%
    with open(NEURON_COUNT_PATH, 'w') as file:
        json.dump(neuron_counts, file)

end_time = time.time()
print('Total time:', end_time - start_time, 'seconds')

key_set = set()
for item in neuron_counts:
    for key in list(item['token_counts'].keys()):
        key_set.add(key)

# ['$', "''", 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
# UD UPOS
# ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
# UD XPOS
# ["''", '-LRB-', '-RRB-', '-RSB-', '.', ':', 'ADJ', 'ADP', 'ADV', 'AUX', 'CC', 'CCONJ', 'CD', 'DET', 'DT', 'EX', 'FW', 'IN', 'INTJ', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NOUN', 'NUM', 'PART', 'PDT', 'POS', 'PRON', 'PROPN', 'PRP', 'PRP$', 'PUNCT', 'RB', 'RBR', 'RBS', 'RP', 'SCONJ', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB', 'WDT', 'WP', 'WP$', 'WRB', 'X', '``']
print(sorted(key_set))
key_list = ["''", '-LRB-', '-RRB-', '-RSB-', '.', ':', 'ADJ', 'ADP', 'ADV', 'AUX', 'CC', 'CCONJ', 'CD', 'DET', 'DT',
            'EX', 'FW', 'IN', 'INTJ', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NOUN', 'NUM', 'PART',
            'PDT', 'POS', 'PRON', 'PROPN', 'PRP', 'PRP$', 'PUNCT', 'RB', 'RBR', 'RBS', 'RP', 'SCONJ', 'SYM', 'TO', 'UH',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB', 'WDT', 'WP', 'WP$', 'WRB', 'X', '``']

neuron_nums = []
for num in range(128):
    neuron_nums.append('Neuron ' + str(num))
# print(neuron_nums)

key_per_neuron = {"''": [], '-LRB-': [], '-RRB-': [], '-RSB-': [], '.': [], ':': [], 'ADJ': [], 'ADP': [], 'ADV': [],
                  'AUX': [], 'CC': [], 'CCONJ': [], 'CD': [], 'DET': [], 'DT': [],
                  'EX': [], 'FW': [], 'IN': [], 'INTJ': [], 'JJ': [], 'JJR': [], 'JJS': [], 'LS': [], 'MD': [],
                  'NN': [], 'NNP': [], 'NNPS': [], 'NNS': [], 'NOUN': [], 'NUM': [], 'PART': [],
                  'PDT': [], 'POS': [], 'PRON': [], 'PROPN': [], 'PRP': [], 'PRP$': [], 'PUNCT': [], 'RB': [],
                  'RBR': [], 'RBS': [], 'RP': [], 'SCONJ': [], 'SYM': [], 'TO': [], 'UH': [],
                  'VB': [], 'VBD': [], 'VBG': [], 'VBN': [], 'VBP': [], 'VBZ': [], 'VERB': [], 'WDT': [], 'WP': [],
                  'WP$': [], 'WRB': [], 'X': [], '``': []}
for neuron in neuron_counts:
    for key in key_list:
        if key not in neuron['token_counts']:
            key_per_neuron[key].append(0)
        else:
            key_per_neuron[key].append(neuron['token_counts'][key])
# print(key_per_neuron)

# %%

# Values
trace_x = []
# Labels
trace_y = []

plotly_tsne = []
for key in key_list:
    trace_neuron = {
        'name': key,
        'x': key_per_neuron[key],
        'y': neuron_nums,
        'orientation': 'h',
        'type': 'bar'
    }
    plotly_tsne.append(trace_neuron)

plotly_tsne_as_json = pd.Series(plotly_tsne).to_json(orient='values')
with open(POS_PER_NEURON_PATH, 'w') as file:
    json.dump(plotly_tsne_as_json, file)

import json
import pickle

from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE

from loading_preprocessing_TC import *

DATASET_PATH = 'resources/datasets/semeval/train/'
files = [DATASET_PATH + 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
         DATASET_PATH + 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml']
train_xml = read_xml(files)
train, answer_texts_train = xml2dataframe_Labels(train_xml, 'train')
answer_texts_train.set_index('answer_id', drop=False, inplace=True)

qa_pairs = train
print(qa_pairs)
answer_texts = answer_texts_train
print(len(answer_texts))

MAX_LENGTH = 200

DATA_PATH = 'out/data/semeval/'
with open(DATA_PATH + 'tokenizer.p', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(DATA_PATH + 'embedding_matrix.p', 'rb') as handle:
    embeddings = pickle.load(handle)


def prepare_data(texts):
    """Tokenize texts and pad resulting sequences of words using Keras functions."""
    global tokenizer, embeddings
    tokens = tokenizer.texts_to_sequences(texts)
    padded_tokens = pad_sequences(tokens, maxlen=MAX_LENGTH, value=embeddings.shape[0] - 1)
    return tokens, padded_tokens


perplexity = 20
print(len(qa_pairs))
all_questions = []
for i in range(len(qa_pairs)):
    current_row = qa_pairs.iloc[i]
    question = current_row['question']
    q_tokens, q_padded_tokens = prepare_data([question])
    all_questions.append(q_padded_tokens[0])
print('TSNE on all questions...')
tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23,
                  metric="cosine")
new_values = tsne_model.fit_transform(all_questions)
print('Finished!')
print(new_values)

trace_question_x = []
trace_question_y = []
trace_question_text = []
trace_question_hovertext = []

for i in range(len(qa_pairs)):
    question = qa_pairs.iloc[i]['question']
    trace_question_x.append(new_values[i][0])
    trace_question_y.append(new_values[i][1])
    trace_question_text.append('Q' + str(i))
    trace_question_hovertext.append(question if len(question) < 61 else question[:60] + '...')

marker_blue = {
    'size': 20,
    'color': 'rgb(0, 0, 255)',
    # star
    'symbol': 17
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

plotly_tsne = [trace_question]
plotly_tsne_as_json = pd.Series(plotly_tsne).to_json(orient='values')

print('Writing to file...')
with open('out/data/plotly_all_questions_json_string_p20.json', 'w') as outfile:
    json.dump(plotly_tsne_as_json, outfile)
    print('Finished!')

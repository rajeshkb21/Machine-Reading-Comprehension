"""
Parses training, development, and testing json then outputs dictionaries containing their structured data.
"""

import json
import pickle


# Saves input python data structure as pickle file in project root
def save_file(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_file(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def main():
    with open("training.json", "r") as read_file:
        data = json.load(read_file)
    with open("development.json", "r") as new_file:
        development_data = json.load(new_file)
    with open("testing.json", "r") as test_file:
        test_data = json.load(test_file)

    data_values = data['data']
    full_dictionary = {}  # Dictionary Holder
    current_value = 0
    question_corpus = []
    impossiblity_corpus = []  # first have to iterate through each of the data values
    for value in data_values:
        paragraph_value = value['paragraphs']
        length_of_paragraphs = len(paragraph_value)
        for num in range(length_of_paragraphs):
            entry_value = paragraph_value[num]
            context_value = entry_value['context']
            qas_value = entry_value['qas']
            length_of_qas_value = len(qas_value)
            for qas_num in range(length_of_qas_value):
                current_question_value = qas_value[qas_num]
                question_val = current_question_value['question']
                id_val = current_question_value['id']
                answer_val = current_question_value['answers']
                impossible_val = current_question_value['is_impossible']
                full_dictionary[current_value] = {'context': context_value, 'question': question_val, 'id': id_val,
                                                  'answer': answer_val, 'impossible_val': impossible_val}
                question_corpus.append(question_val)
                if (impossible_val):
                    impossiblity_corpus.append(0)
                else:
                    impossiblity_corpus.append(1)
                current_value += 1

    development_data_values = development_data['data']
    development_full_dictionary = {}
    development_question_corpus = []
    development_impossiblity_corpus = []
    for value in development_data_values:
        paragraph_value = value['paragraphs']
        length_of_paragraphs = len(paragraph_value)
        for num in range(length_of_paragraphs):
            entry_value = paragraph_value[num]
            context_value = entry_value['context']
            qas_value = entry_value['qas']
            length_of_qas_value = len(qas_value)
            for qas_num in range(length_of_qas_value):
                current_question_value = qas_value[qas_num]
                question_val = current_question_value['question']
                id_val = current_question_value['id']
                answer_val = current_question_value['answers']
                impossible_val = current_question_value['is_impossible']
                development_full_dictionary[current_value] = {'context': context_value, 'question': question_val,
                                                              'id': id_val, 'answer': answer_val,
                                                              'impossible_val': impossible_val}
                development_question_corpus.append(question_val)
                if (impossible_val):
                    development_impossiblity_corpus.append(0)
                else:
                    development_impossiblity_corpus.append(1)
                current_value += 1

    test_data_values = test_data['data']
    test_full_dictionary = {}
    test_question_corpus = []
    test_impossiblity_corpus = []
    test_id_corpus = []
    for value in test_data_values:
        paragraph_value = value['paragraphs']
        length_of_paragraphs = len(paragraph_value)
        for num in range(length_of_paragraphs):
            entry_value = paragraph_value[num]
            context_value = entry_value['context']
            qas_value = entry_value['qas']
            length_of_qas_value = len(qas_value)
            for qas_num in range(length_of_qas_value):
                current_question_value = qas_value[qas_num]
                question_val = current_question_value['question']
                id_val = current_question_value['id']
                test_full_dictionary[current_value] = {'context': context_value, 'question': question_val, 'id': id_val}
                test_question_corpus.append(question_val)
                test_id_corpus.append(id_val)
                current_value += 1

    TRAINING_DICT = "training_dictionary.pickle"
    DEV_DICT = "development_dictionary.pickle"
    TEST_DICT = "test_dictionary.pickle"
    save_file(TRAINING_DICT, full_dictionary)
    save_file(DEV_DICT, development_full_dictionary)
    save_file(TEST_DICT, test_full_dictionary)


if __name__ == '__main__':
    main()

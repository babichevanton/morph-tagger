#coding=CP1251

import cPickle
import copy
from math import log
import nltk


class Solution(nltk.tag.api.TaggerI):
    def __init__(self, train_data):
        stemmer = nltk.stem.snowball.RussianStemmer()

        # tag_nexttag is a dict, where keys are tags and values are pairs of dicts with next tags and nums of entries
        self.tag_nexttag = {}
        # word_tag is a dict, where keys are words and values are pairs of dicts with tags and nums of entries
        self.word_tag = {}

        #initialize blank first tag
        self.tag_nexttag[' '] = [{}, 0]
        previous = ' '

        for sent in train_data:
            for token in sent:
                word = stemmer.stem(token[0].lower().decode('CP1251'))
                tag = token[1]
                # Filling dictionary by records about 'previous-tag' - 'tag'
                previous_line = self.tag_nexttag[previous]
                previous_line[1] += 1
                if tag in previous_line[0]:
                    previous_line[0][tag] += 1
                else:
                    previous_line[0][tag] = 1
                if not (tag in self.tag_nexttag):
                    self.tag_nexttag[tag] = [{}, 0]
                # Filling dictionary by records about 'word' - 'tag'
                if word in self.word_tag:
                    self.word_tag[word][1] += 1
                    if tag in self.word_tag[word][0]:
                        self.word_tag[word][0][tag] += 1
                    else:
                        self.word_tag[word][0][tag] = 1
                else:
                    self.word_tag[word] = [{}, 1]
                    self.word_tag[word][0][tag] = 1
                previous = tag

        del self.tag_nexttag[' ']

    @staticmethod
    def __view_dictionary(dictionary):
        for key in dictionary.keys():
            pair = dictionary[key]
            print key, pair[1]
            tag_info = []
            for sub_key in pair[0].keys():
                tag_info.append(sub_key + ": " + str(pair[0][sub_key]))

            print "    " + (", ".join(tag_info))

    def view_tables(self):
        self.__view_dictionary(self.tag_nexttag)
        print "\n"
        self.__view_dictionary(self.word_tag)

    def tag(self, tokens):
        stemmer = nltk.stem.snowball.RussianStemmer()
        tokens = map(lambda x: stemmer.stem(x.lower().decode('CP1251')), tokens)
        # creating begin-list of tags. All probabilities are 1/num_of_tags
        tags = self.tag_nexttag.keys()
        probability = log(1.0 / len(tags))
        previous_dict = {}
        for cur_tag in tags:
            previous_dict[cur_tag] = probability

        # Viterbi algorythm
        result = []
        for token in tokens:
            current_dict = {}
            tag_max_prob = -100000.0
            result_tag = ''
            for cur_tag in tags:
                if token in self.word_tag.keys():
                    cur_word = self.word_tag[token]
                    if cur_tag in cur_word[0].keys():
                        tag_prob = log(float((cur_word[0][cur_tag] + 1)) / (cur_word[1] + len(tags)))
                    else:
                        tag_prob = log(1.0 / (cur_word[1] + len(tags)))
                else:
                    tag_prob = 0.0
                max_prob = -100000.0
                for prev_tag in tags:
                    prev_tag_info = self.tag_nexttag[prev_tag]
                    if cur_tag in prev_tag_info[0].keys():
                        prev_tag_prob = log(float((prev_tag_info[0][cur_tag] + 1)) / (prev_tag_info[1] + len(tags)))
                    else:
                        prev_tag_prob = log(1.0 / (prev_tag_info[1] + len(tags)))
                    prev_tag_prob += previous_dict[prev_tag]
                    if prev_tag_prob > max_prob:
                        max_prob = prev_tag_prob

                tag_prob += max_prob
                current_dict[cur_tag] = tag_prob
                if tag_prob > tag_max_prob:
                    tag_max_prob = tag_prob
                    result_tag = cur_tag

            result.append(result_tag)
            previous_dict = copy.deepcopy(current_dict)

        return result


def load_train(filename):
    input_file = open(filename, 'r')
    train_data = cPickle.load(input_file)
    input_file.close()

    return Solution(train_data)


#load_train('train.pkl').view_tables()

#text = 'Стекло время . Стекло разбилось .'
#words = nltk.PunktWordTokenizer().tokenize(text)
#tags = load_train('train.pkl').tag(words)


#for word, tag in zip(words, tags):
#    print word.decode('CP1251'), tag
import tensorflow as tf
import numpy as np
from ordered_set import OrderedSet



data_predicted = "C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/data/programGeneratedData/768remainingtestdata2016.txt" #This is the data that I want to predict
embedding_file = "C:/Users/Kelvin/Desktop/Thesis/Prior Papers/HAABSA_plusplus/venv/data/programGeneratedData/768embedding2016.txt" #The fine tuned BERT model
model_path = 'C:\\Users\\Kelvin\\Desktop\\Thesis\\Prior Papers\\HAABSA_plusplus\\venv\\trainedModelMaria\\2016\\Best\\-188' #The model graph file

def model_predictions(data_predicted):
    model = 'Maria'
    f = classifier(data_predicted)
    dict = f.get_Allinstances()
    x_left = dict['x_left']
    x_left_len = dict['x_left_len']
    x_right = dict['x_right']
    x_right_len = dict['x_right_len']
    y_true = dict['y_true']
    target_word = dict['target']
    target_words_len = dict['target_len']
    size_1 = dict['size']
    size_2 = dict['size']

    predictions, probabilities = f.get_allProb(x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, size_1, size_2)
    return probabilities


def orderProb(probabilities):
    """
    convert probs from the classifier into the right format: [neg,neu,pos]
    """
    nr, nc = probabilities.shape
    correctProb = np.zeros((nr, nc))

    correctProb[:, 1] = probabilities[:, 2]
    correctProb[:, 2] = probabilities[:, 0]
    correctProb[:, 0] = probabilities[:, 1]

    return correctProb


def load_w2v(w2v_file, embedding_dim=768, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v


def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    word_to_id = word_id_file #word_id_file is the list with words (including token number) and its corresponding row number in the embedding file
    print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    # read in txt file
    lines = open(input_file).readlines() #open data_predicted file and makes a list of strings per line. Ends with \n after each line to indicate new line
    for i in range(0, len(lines), 3): #len(lines) = amount of lines in the text file. Currently 248*3 lines
        # targets
        words = lines[i + 1].lower().split() #creates a list of all the (sub)words in the aspect
        target = words

        target_word = []
        for w in words: #amount of words in the aspect
            if w in word_to_id:
                target_word.append(word_to_id[w]) #appends the line/row numbers of the (sub)words in the aspect
        l = min(len(target_word), target_len) #l = the amount of (sub)words in the aspect, up to a maximum of 10
        tar_len.append(l) #tar_len is a list of amount of sub(words) in each aspect, up to a maximum of 10
        target_words.append(target_word[:l] + [0] * (target_len - l)) #target_words is a list of the line/row numbers of each sub(word) in the aspect; the list consists of 10 numbers, with the remaining numbers becoming 0 if there are no more words left in the aspect

        # sentiment
        y.append(lines[i + 2].strip().split()[0]) #list of sentiments

        # left and right context
        words = lines[i].lower().split() #a list of the (sub)words, split between each (sub)word
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word]) #appends all the line/row numbers of the (sub)words that come before $t$, aka the left part
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word]) #appends all the line/row numbers of the (sub)words that come after $t$, aka the right part
        # words_l.extend(target_word)
        words_l = words_l[:sentence_len] #sentence_len = 80, the first 80 (sub)word tokens in the left part
        words_r = words_r[:sentence_len] #sentence_len = 80, the first 80 (sub)word tokens in the right part
        sen_len.append(len(words_l)) #the length of the left sentences appended in a list
        x.append(words_l + [0] * (sentence_len - len(words_l))) #x is a list of the line/row numbers of the (sub)words in the left part, with maximum 80; less than 80 means the rest is filled with zero's
        # tmp = target_word + words_r
        tmp = words_r #the first 80 (sub)word tokens in the right part
        tmp.reverse() #Reverses the order of the list of tokens in the right part
        sen_len_r.append(len(tmp)) #the length of the right sentences appended in a list
        x_r.append(tmp + [0] * (sentence_len - len(tmp))) #x_r is a list of the line/row numbers of the (sub)words in the right part, with maximum 80; less than 80 means the rest is filled with zero's
        all_sent.append(sent) #a list of a list of all sub(words) of the sentence
        all_target.append(target) #a list of a list of all sub(words) of the aspect

    all_y = y; #list of all sentiments ['1', '1', ... , ]
    y = change_y_to_onehot(y) #array of sentiment labels, as a list of three [[1, 0, 0], [1, 0, 0], [1, 0, 0], ... ] but listed vertically


    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC': #TC currently
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(all_target), np.asarray(all_y)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def change_y_to_onehot(y):
    from collections import Counter
    print(Counter(y)) #Counter({'1': 144, '-1': 82, '0': 22})
    class_set = OrderedSet(['1','-1','0']) #Positive, negative, neutral
    n_class = len(class_set) #3
    y_onehot_mapping = dict(zip(class_set, range(n_class))) #{'1': 0, '-1': 1, '0': 2}
    print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class #temporary list of zero's [0, 0, 0]
        tmp[y_onehot_mapping[label]] = 1 #temporary list of sentiment labels e.g. [1, 0, 0] for positive
        onehot.append(tmp) #append the sentiment to onehot; results in a list of a list of sentiment labels e.g. [[1, 0, 0], [1, 0, 0], [1, 0, 0], ... ]
    return np.asarray(onehot, dtype=np.int32) #[[1, 0, 0], [1, 0, 0], [1, 0, 0], ... ] but as an array (list with all labels listed vertically)


def getStatsFromFile(path):
    polarity_vector = []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 3
        for i in range(0, len(lines), 3):
            # polarity
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector


class classifier:

    def __init__(self,  input_file):
        """
        Constructor to initialize a black box model
        :param input_file: the file containing the .txt data of the instances (reviews)
        :param model_path: the path to the trained model
        """
        input_file = data_predicted
        embedding = embedding_file
        model = "Maria"

        self.input_file = input_file
        self.model_path = 'C:\\Users\\Kelvin\\Desktop\\Thesis\\Prior Papers\\HAABSA_plusplus\\venv\\trainedModelMaria\\2016\\Best\\-188'  #model graph

        self.model = model
        # \\programGeneratedData\\" + "small_" + str(
        self.word_id_mapping, self.w2v = load_w2v(embedding, 768)   #Word_id_mapping = a dictionary with each word in embedding file (including number) and its corresponding row (e.g. 'positive_6': 44601)
                                                                    #w2v = a dictionary with the word vector for each corresponding row in the embedding file

        #x_left = array of a list of the line/row numbers of the (sub)words in the left part, with maximum 80; less than 80 means the rest is filled with zero's
        #x_left_len = array of the length of the left sentences appended in a list
        #x_right = array of a list of the line/row numbers of the (sub)words in the right part, with maximum 80; less than 80 means the rest is filled with zero's
        #x_right_len = array of the length of the right sentences appended in a list
        #y_true = array of sentiment labels, as a list of three [[1, 0, 0], [1, 0, 0], [1, 0, 0], ... ] but listed vertically
        #target_word = array of a list of the line/row numbers of each sub(word) in the aspect; the list consists of 10 numbers, with the remaining numbers becoming 0 if there are no more words left in the aspect
        #target_words_len = array of a list of amount of sub(words) in each aspect, up to a maximum of 10
        #_ = all_sent = array of a list of a list of all sub(words) of the sentence
        #_ = all_target = a list of a list of all sub(words) of the aspect
        #_ = all_y = list of all sentiments ['1', '1', ... , ]


        self.x_left, self.x_left_len, self.x_right, self.x_right_len, self.y_true, self.target_word, \
        self.target_words_len, _, _, _ = load_inputs_twitter(self.input_file, self.word_id_mapping, #load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8')
                                                             80, #80 = max_sentence_len, max number of tokens per sentence
                                                             'TC', '1', 19) #19 = max target length

        self.size, self.polarity = getStatsFromFile(self.input_file) #Get total amount of sentences (lines / 3) and the polarity vector
        #polarity are the correct labels, given in data

        self.size = int(self.size)
        # restoring the trained model
        # delete the current graph
        tf.reset_default_graph()
        # import the loaded graph
        self.imported_graph = tf.train.import_meta_graph(self.model_path + '.meta')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # restore the saved variables
        self.imported_graph.restore(self.sess, self.model_path)
        self.graph = tf.get_default_graph()

        # all_nodes = [n for n in self.graph.as_graph_def().node]
        # print(all_nodes) # visualise nodes in the graph to re-do tensor mapping

        # inversez x cu x_bw si x_len cu x_bw_len
        # setting keys for the feed dict
        self.x = self.graph.get_tensor_by_name('inputs/Placeholder:0')
        self.y = self.graph.get_tensor_by_name('inputs/Placeholder_1:0')
        self.sen_len = self.graph.get_tensor_by_name('inputs/Placeholder_2:0')
        self.x_bw = self.graph.get_tensor_by_name('inputs/Placeholder_3:0')
        self.sen_len_bw = self.graph.get_tensor_by_name('inputs/Placeholder_4:0')
        self.target_words = self.graph.get_tensor_by_name('inputs/Placeholder_5:0')
        self.tar_len = self.graph.get_tensor_by_name('inputs/Placeholder_6:0')
        self.keep_prob1 = self.graph.get_tensor_by_name('Placeholder:0')
        self.keep_prob2 = self.graph.get_tensor_by_name('Placeholder_1:0')
        self.prob = self.graph.get_tensor_by_name('softmax/Softmax:0')


    def get_Allinstances(self):
        """
        method to return all instances in a dictionary
        :return:
        """
        correctSize = 0

        predictions, probabilities = self.get_allProb(self.x_left, self.x_left_len,
                                                      self.x_right, self.x_right_len,
                                                      self.y_true,
                                                      self.target_word,
                                                      self.target_words_len,
                                                      self.size,
                                                      self.size)

        correctDict = {
            'x_left': [],
            'x_left_len': [],
            'x_right': [],
            'x_right_len': [],
            'target': [],
            'target_len': [],
            'y_true': [],
            'true_label': [],
            'pred': []
        }

        for i in range(int(self.size)):
            correctDict['x_left'].append(self.x_left[i])
            correctDict['x_right'].append(self.x_right[i])
            correctDict['x_left_len'].append(self.x_left_len[i])
            correctDict['x_right_len'].append(self.x_right_len[i])
            correctDict['target'].append(self.target_word[i])
            correctDict['target_len'].append(self.target_words_len[i])
            correctDict['y_true'].append(self.y_true[i])
            correctDict['true_label'].append(int(self.polarity[i]))
            correctDict['pred'].append(predictions[i])
            correctSize += 1
        correctDict['size'] = correctSize

        return correctDict


    def get_allProb(self, x_left, x_left_len, x_right, x_right_len, y_true, target_word, target_words_len, batch_size,
                    num_samples):
        """
        Almost the same as get_prob, but here we input all instances at the same time and get a probability matrix
        and prediction vector. Input are arrays/matrices with as row length the sample size.
        """
        probabilities = np.zeros((num_samples, 3))
        predictions = np.zeros(num_samples)

        for i in range(int(num_samples / batch_size)):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size

            feed_dict = {
                self.x: x_left[batch_start:batch_end], # x_left is also x_fw or x
                self.x_bw: x_right[batch_start:batch_end],
                self.y: y_true[batch_start:batch_end],
                self.sen_len: x_left_len[batch_start:batch_end], # sen_len is also sen_len_fw
                self.sen_len_bw: x_right_len[batch_start:batch_end],
                self.target_words: target_word[batch_start:batch_end],
                self.tar_len: target_words_len[batch_start:batch_end],
                self.keep_prob1: 1,
                self.keep_prob2: 1,
            }

            ##getting prediction of instance

            prob = self.sess.run(self.prob, feed_dict=feed_dict) # prob matrix for all instances for the sentim (n_inst x 3)
            prob = orderProb(prob) # convert probs from the classifier into the right format: [neg,neu,pos]

            pred = np.argmax(prob, axis=1) - 1 # highest prob class chosen and coverted from 0, 1, 2 to corr sent

            probabilities[batch_start:batch_end, :] = prob
            predictions[batch_start:batch_end] = pred

        return predictions, probabilities


print(model_predictions(data_predicted))


# -*- coding: utf-8 -*-
'''
main函数
create time：2018年10月25日11:17:30
author:joe hammer
'''
import re
import os
import logging
import sys
import multiprocessing
import time

import numpy
from numpy import *

from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from nltk.stem import SnowballStemmer

import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块

import Phrase_LDA.phrase_mining as phm
import Phrase_LDA.utils as utils

import stanfordcorenlp

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim.models


min_support=20      #保留的最小词频的vocab
alpha=5             #两个单词合并的阈值，一般取值3-5，越好获取的词组质量越好，但是数量越少
max_phrase_size=6   #组成词组的word最大数量，决定一个词组最多由多少个单词组成

default_min_df=5    #某个词的document frequence小于min_df，则这个词不会被统计，如：tfidf,word2vec...
default_max_df=0.7  #可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。
                    # 这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。
                    # 如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。

def phraseSFilter(sentence,phraseList):
    """
    words to phrases "word1-word2". and lower
    """
    # tmpSentence = sentence.lower() #带符号
    sentence = re.sub('[^A-Za-z0-9-]+', ' ', sentence.lower())
    tmpSentence = sentence #不带符号
    words = re.split(" ", sentence.lower())
    # print(len(words))
    # print(phraseArray)
    if len(words) > max_phrase_size:
        # print(max_phrase_size)
        for i in range(0,len(words)-1):
            if len(words) - i  > max_phrase_size:
                for j in range(i + max_phrase_size, i + 1, -1):
                    tmp_phr = ' '.join(words[i:j])
                    # print(' '.join(words[i:j]))
                    # if ['true'] == ["true" for k, v in enumerate(phraseArray) if v[0] == tmp_phr]:
                    if ['true'] == ["true" for v in phraseList if v == tmp_phr]:
                        tmp_phr_re = tmp_phr.replace(" ", "-")
                        tmpSentence = tmpSentence.replace(tmp_phr, tmp_phr_re)
            else:
                for j in range(len(words), i + 1, -1):
                    tmp_phr = ' '.join(words[i:j])
                    # print(' '.join(words[i:j]))
                    if ['true'] == ["true" for v in phraseList if v == tmp_phr]:
                        tmp_phr_re = tmp_phr.replace(" ", "-")
                        tmpSentence = tmpSentence.replace(tmp_phr, tmp_phr_re)
    else:
        for i in range(0,len(words)-1):
            for j in range(len(words), i+1, -1):
                tmp_phr = ' '.join(words[i:j])
                if ['true'] == ["true" for v in phraseList if v == tmp_phr]:
                    tmp_phr_re = tmp_phr.replace(" ", "-")
                    tmpSentence = tmpSentence.replace(tmp_phr, tmp_phr_re)

    return tmpSentence

def stemFilter(sentence):
    """
    stemFilter
    """
    sentence = re.sub('[^A-Za-z0-9-]+', ' ', sentence.lower())
    words = re.split(" ", sentence)
    new_sentence = ""
    for word in words:
        newword = SnowballStemmer("english").stem(word)
        new_sentence = new_sentence + " " + newword
    return new_sentence

def stopwordFilter(sentence,stopword_list):
    """
    stopwordFilter
    """
    words = re.split(" ", sentence.lower())
    new_sentence = ""
    for word in words:
        if word not in stopword_list:
            new_sentence = new_sentence + " " + word
    return new_sentence.strip()

def get_tf(org_line,min_df):
    """
    计算tf值
    TF = 在某一类中词条w出现的次数 / 该类中所有的词条数目
    text:该词所在文档
    filename：类似于max_df，不同之处在于如果某个词的document frequence小于min_df，则这个词不会被当作关键词
    return:dict word_tf(该文本出现的词的tf值)
    """
    word_freq = {}  # 词频dict
    word_tf = {}  # 词的tf值dict

    # org_line = load_data(filename)
    # get word
    orgString =" ".join(org_line) #list to string
    tmpSpliter = re.split(" ", orgString.lower()) #分词
    idSpliter = list(set(tmpSpliter)) #去重

    tmpStr = ""
    for i in range(0, len(idSpliter)):
        # print(str(i) + " / " + str(len(idSpliter)) + ": tf of " + idSpliter[i] + " is processing...")
        tmpword = idSpliter[i]
        tmp_freq = tmpSpliter.count(tmpword)
        if tmpword != " " and tmp_freq >=min_df:
            word_freq[tmpword] = tmpSpliter.count(tmpword)
            rs_float = word_tf[tmpword] = float(tmpSpliter.count(tmpword) / len(idSpliter))
            print(str(i) + " / " + str(len(idSpliter)) + ": tf of " + idSpliter[i] + " is processed, result: " + str(rs_float))

    word_freq = sorted(word_freq.items(), key=lambda d: d[1], reverse=True)
    word_tf = sorted(word_tf.items(), key=lambda d: d[1], reverse=True)

    return word_freq, word_tf


def get_idf(org_line, max_df):
    """
    计算idf值
    IDF = log(语料库的文档总数 / (包含词w的文档数 + 1))           +1是为了防止分母为0
    word：要计算的词
    filename:包含所有语料的list，一个文件为其中一个元素
    max_df:可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。
        这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。
        如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。
        如果参数中已经给定了vocabulary，则这个参数无效
    return:该词的idf值
    """
    word_idf = {}  # 词的idf值dict
    # org_line = load_data(filename)
    num_corpus = len(org_line)  # 文章个数
    # print(num_corpus)
    # get word
    orgString = " ".join(org_line)  # list to string
    tmpSpliter = re.split(" ", orgString.lower())  # 分词
    idSpliter = list(set(tmpSpliter))  # 去重
    # print(idSpliter)

    tmpStr = ""
    for i in range(0, len(idSpliter)):

        tmpword = idSpliter[i]
        if tmpword != " " :
            count = 0
            for cur_corpus in org_line:
                if tmpword in set(re.split(" ", cur_corpus.lower())):
                    count += 1
            # print(tmpword,count)
            tmp_df_percent = float( count/ (num_corpus + 1))
            # print(tmp_df_percent)
            if tmp_df_percent <= max_df:
                rs_float = word_idf[tmpword] = math.log(float(num_corpus / (count + 1)))
                print(str(i) + " / " + str(len(idSpliter)) + ": idf of " + idSpliter[i] + " is processed, result: " + str(rs_float))

    word_idf = sorted(word_idf.items(), key=lambda d: d[1], reverse=True)

    return word_idf

def get_tfidf(filename, max_df, min_df, stop_words="input/stopwords.txt"):
    """
    计算tfidf值
    filename:包含所有语料的list，一个文件为其中一个元素
    return:该词的tfidf值
    """
    word_tfidf = {}  # 词的tfidf值dict
    org_line = utils.load_data(filename)
    word_freq, word_tf = get_tf(org_line,min_df)
    word_idf = get_idf(org_line, max_df)
    i = 1
    for key, values in word_idf:
        stopword_list = [line.strip() for line in open(stop_words, 'r', encoding='utf-8').readlines()]
        if key not in stopword_list:
            for tfkey, tfvalues in word_tf:
                if tfkey == key:
                    tfidfvalue=tfvalues*values
                    word_tfidf[key] = tfidfvalue
                    print(str(i) + " / " + str(len(word_idf)) + ": tf-idf of " + key + " is processed, result: " + str(tfidfvalue))
        i = i +1

    word_tfidf = sorted(word_tfidf.items(), key=lambda d: d[1], reverse=True)
    # utils.store_file(word_tfidf, file_mytfidf)
    return word_tfidf


if __name__=="__main__":

    # 0、准备
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    file_orginal = "input/train_input.data"
    file_stemmer = "input/input_stmer.data"
    file_stopword = "input/stopwords.txt"
    file_phrases = "intermediate_output/phrases.txt"
    file_wp = "intermediate_output/phrases_train_input.txt"
    # file_wp = "input/input_stmer.data"
    file_word2vec_model = "intermediate_output/tmword2vec.txt"
    file_word2vec_txt_vec_orginal = "intermediate_output/tmword2vec.txt.vector"
    file_word2vec_vec = "intermediate_output/tmword2vec.vector"
    file_word2vec_1vec = "intermediate_output/tmword2vec_one.vector"
    file_word2vec_2vec = "intermediate_output/tmword2vec_two.vector"
    file_2vec_classifier = "intermediate_output/vec_classifier.vector"
    file_vec_tfidf = "intermediate_output/vec_tfidf.vector"
    file_mytfidf = "intermediate_output/mytfidf.vector"

    # 0、文档预处理，词干处理
    if not os.path.exists(file_stemmer):
        print('=== 未检测到有"file_stemmer"短语文件存在，开始词干处理 ===')
        org_line = utils.load_data(file_orginal)
        # stopwords = [line.strip() for line in open(file_stopword, 'r', encoding='utf-8').readlines()]
        _txt_stemmer = []
        n_phr_processor = 1
        for line in org_line:
            print("Step 0, ", n_phr_processor, " / ", len(org_line), " : ", line[0:23], " ... ")
            tmpLine = line
            # tmpLine = stopwordFilter(tmpLine, stopwords)
            tmpLine = stemFilter(tmpLine)
            _txt_stemmer.append(tmpLine)
            n_phr_processor = n_phr_processor + 1
        print("0. store file_stemmer ....")
        utils.store_file(_txt_stemmer, file_stemmer)
        print('=== 0.F. 词干处理完成 ===')
    else:
        print('=== 检测到"file_stemmer"短语文件存在，跳过该阶段 ===')


    # 1、单词合并成短语  composed words
    if not os.path.exists(file_phrases):
        print('=== 未检测到有"file_phrases"短语文件存在，开始合成短语 ===')
        phrase_miner = phm.PhraseMining(file_stemmer, min_support=min_support,alpha=alpha,max_phrase_size=max_phrase_size)
        partitioned_docs, index_vocab = phrase_miner.mine()
        frequent_phrases = phrase_miner.get_frequent_phrases()
        # utils.store_partitioned_docs(partitioned_docs)
        # utils.store_vocab(index_vocab)
        utils.get_store_phrases(frequent_phrases) # intermediate_output/phrases.txt
        # print("1.F. phrase_miner finished ....")
        print('=== 1.F. 合成短语完成 ===')
    else:
        print('=== 检测到"file_phrases"短语文件存在，跳过该阶段 ===')

    # 2、输入文档预处理，包括：标识出短语
    if not os.path.exists(file_wp):
        print('=== 未检测到有"file_wp"输入文档存在，开始输入文档预处理，标识出短语 ===')
        org_line = utils.load_data(file_stemmer)
        file_phrases = "intermediate_output/phrases.txt"
        phr_list = utils.load_data(file_phrases)
        _txt_wp = []
        print("2.1. phraseSFilter ....")
        n_phr_processor = 1
        for line in org_line:
            print("Step 2.1, ", n_phr_processor," / ",len(org_line)," : ", line[0:random.randint(29,49)]," ... ")
            tmpLine = phraseSFilter(line, phr_list)
            _txt_wp.append(tmpLine)
            n_phr_processor = n_phr_processor +1
        print("2.2. store_phraseFile ....")
        utils.store_phraseFile(_txt_wp) #intermediate_output/phrases_train_input.txt
        # print("2.F. phraseSFilter finished ....")
        print('=== 2.F. 输入文档预处理，标识出短语，完成 ===')
    else:
        print('=== 检测到"file_wp"输入文档存在，跳过该阶段 ===')


    # 3、word2vec, intermediate_output/tmword2vec.txt.vector
    if not os.path.exists(file_word2vec_txt_vec_orginal):
        print('=== 未检测到有"file_word2vec_txt_vec_orginal"输入文档存在，开始word2vec训练 ===')
        # sg=1 skip-gram; sg=0 CBOW size=50, window=5, min_count=5
        model = Word2Vec(LineSentence(file_wp), sg=1, size=50, window=5, min_count=default_min_df,
                         workers=multiprocessing.cpu_count())
        # trim unneeded model memory = use(much) less RAM
        # model.init_sims(replace=True)
        model.save(file_word2vec_model)
        model.wv.save_word2vec_format(file_word2vec_txt_vec_orginal, binary=False)
        print('=== 3.F. word2vec训练完成 ===')
    else:
        print('=== 检测到"file_word2vec_txt_vec_orginal"输入文档存在，跳过该阶段 ===')

    # 3.1 word2vec reduction dimensions 降维处理, intermediate_output/tmword2vec_one.vector
    if not os.path.exists(file_word2vec_2vec):
        print('=== 未检测到有"file_word2vec_2vec"文档存在，开始降维处理 ===')
        txtvector_sent_line = utils.load_data(file_word2vec_txt_vec_orginal)
        _vecter_sents_line = []
        _word_list = []
        for line in txtvector_sent_line:
            if len(line) > 30:
                # tmpLine = re.sub('[^0-9.-]+', ' ', line.lower()).strip()
                tmpSpliter = re.split(" ", line.lower())
                tmpStr = ""
                for i in range(1, len(tmpSpliter)):
                    tmpStr = tmpStr + " " + tmpSpliter[i]
                _vecter_sents_line.append(tmpStr.strip())
                _word_list.append(tmpSpliter[0].strip())
        utils.store_file(_vecter_sents_line, file_word2vec_vec)  # intermediate_output/tmword2vec.vector
        print("3.1 tmword2vec.vector, no words, just vector ....")

        inp_reduc = numpy.loadtxt(file_word2vec_vec, dtype=float32)
        # inp_reduc = utils.load_data(file_word2vec_vec)
        n = 2
        X_tsne = TSNE(n_components=n, learning_rate=100).fit_transform(inp_reduc)
        # PCA reduce the dimension of word vector
        # X_PCA = PCA(n_components=2).fit_transform(inp_reduc)
        utils.store_vecfile(X_tsne, n, _word_list, file_word2vec_2vec)
        # print("3.1 reduction dimensions finished ....")
        print('=== 3.1.F. word2vec降维处理完成 ===')
    else:
        print('=== 检测到"file_word2vec_2vec"文档存在，跳过该阶段 ===')

    # 3.2、word2vec 使用
    # modelwv = gensim.models.Word2Vec.load_word2vec_format(file_word2vec_txt_vec_orginal, binary = True) #DeprecationWarning
    # model = Word2Vec.load(file_word2vec_model)
    # modelwv.most_similar(positive=['woman', 'king'], negative=['man'])    # 输出[('queen', 0.50882536), ...]
    # modelwv.doesnt_match("breakfast cereal dinner lunch".split())    # 输出'cereal'
    # modelwv.similarity('woman', 'man')    # 输出0.73723527
    # modelwv['computer']  # raw numpy vector of a word    # 输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
    # print(model['language'])

    # modelvec = gensim.models.KeyedVectors.load_word2vec_format(file_word2vec_txt_vec_orginal)  # used here
    # print(modelvec['language'])
    modelvec_one = gensim.models.KeyedVectors.load_word2vec_format(file_word2vec_2vec)
    print(modelvec_one['electron'])
    try:
        c = modelvec_one['zyrian']
    except KeyError:
        c = 0
    print(c)
    # for k,v in modelvec_one:
    #     print(k)

    # 4、TFIDF
    if not os.path.exists(file_mytfidf):
        print('=== 未检测到有"file_mytfidf"文档存在，开始TFIDF处理 ===')
        max_df = 0.6
        min_df = 5
        word_tfidf = get_tfidf(file_wp, max_df, min_df,stop_words=file_stopword )
        utils.store_vec_tfidf(word_tfidf, file_mytfidf)
        # print("3.1 reduction dimensions finished ....")
        print('=== 4.F. TFIDF处理完成 ===')
    else:
        print('=== 检测到"file_mytfidf"文档存在，跳过该阶段 ===')


    # 5 save wordvec value of tfidf-top5000

    model2vec = gensim.models.KeyedVectors.load_word2vec_format(file_word2vec_2vec)  # used here
    tfidf_line = utils.load_data(file_mytfidf)
    _2vec_value = []
    for value in tfidf_line:
        tmpSpliter = re.split("\t", str(value))
        # print(tmpSpliter[0])
        try:
            vec_rs = model2vec[tmpSpliter[0]]
        except KeyError:
            vec_other = 0
        _2vec_value.append(vec_rs)

    print('=== 5.F. file_2vec_classifier,分类数据准备完成 ===')



    # 6、classifier
    #
    # 样品数据量
    #
    topN = 4000  # 从语料库的关键词集中提取前topN个词，作为分类的基础词
    vec_line = _2vec_value
    realnumber = 0
    if len(vec_line)>topN:
        realnumber = topN
    else:
        realnumber = (len(vec_line)//100)*100

    print(realnumber)

    # #
    # # 样品数范围，测试
    # #-----------------------------------------------------------
    # # MSE pre-valued
    # y_test = []
    # y_predict = []
    # y_predict_real = 0
    # s_score_max = 0.0
    # stepnumber = 1
    # # begin ---------
    # _tmp_quantile_ = 0.09
    # for i in range(1000, realnumber+1, 1000):
    #     # n_reshape = i//2
    #     tmp_array = vec_line[:i]
    #     _X_orginal = numpy.array(tmp_array).reshape(i, 2)
    #     # print(_X_orginal)
    #
    #     # #
    #     # # mean shift 预测聚类个数
    #     # #
    #     # ##带宽，也就是以某个点为核心时的搜索半径.
    #     # # bandwidth = estimate_bandwidth(_X_orginal, quantile=0.2, n_samples=500)
    #     # bandwidth = estimate_bandwidth(_X_orginal, quantile=_tmp_quantile_)
    #     # # bandwidth = estimate_bandwidth(_X_orginal)
    #     # ##设置均值偏移函数
    #     # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #     # ##训练数据
    #     # ms.fit(_X_orginal)
    #     # ##每个点的标签
    #     # labels = ms.labels_
    #     # # print(labels)
    #     # ##簇中心的点的集合
    #     # cluster_centers = ms.cluster_centers_
    #     # ##总共的标签分类
    #     # labels_unique = numpy.unique(labels)
    #     # ##聚簇的个数，即分类的个数
    #     # n_clusters_ = len(labels_unique)
    #
    #     #
    #     # AffinityPropagation 预测聚类个数
    #     #
    #     # 不足：
    #     # 1、preference参数，AP聚类应用中需要手动指定Preference和Damping factor，这其实是原有的聚类“数量”控制的变体
    #     # 2、算法较慢。由于AP算法复杂度较高，运行时间相对K-Means长，这会使得尤其在海量数据下运行时耗费的时间很多。
    #     # 不实用
    #     # --------------------
    #     apTime_start = time.time()
    #     # ap_algorithm = AffinityPropagation(preference=-50).fit(_X_orginal)
    #     ap_algorithm = AffinityPropagation(preference=-50).fit(_X_orginal)
    #     ap_cluster_centers_indices = ap_algorithm.cluster_centers_indices_
    #     ap_labels = ap_algorithm.labels_
    #     n_clusters_ = len(ap_cluster_centers_indices)
    #     apTime_end = time.time()
    #
    #
    #     if n_clusters_ > 1:
    #         kmeans_model = KMeans(n_clusters=n_clusters_, random_state=1).fit(_X_orginal)
    #         labels = kmeans_model.labels_
    #         s_score = metrics.silhouette_score(_X_orginal, labels, metric='euclidean')
    #         c_score = metrics.calinski_harabaz_score(_X_orginal, labels)
    #     else:
    #         s_score = 0
    #         c_score = 0
    #
    #     # print("TopN %d, number of estimated clusters : %d, Score: %f, %f" % (i, n_clusters_, s_score, c_score))
    #     print(i, n_clusters_, s_score, apTime_end-apTime_start)
    #
    #     #
    #     # MSE pre-valued
    #     #
    #     y_test.append(s_score)
    #     if s_score_max < s_score:
    #         s_score_max = s_score
    #         y_predict_real = s_score
    #     stepnumber = stepnumber + 1
    #
    #
    # #
    # # mse 均方误差
    # # -----------------------------------------
    # for i in range (1,stepnumber):
    #     y_predict.append(y_predict_real)
    # mse_score = metrics.mean_squared_error(y_test, y_predict)# 均方误差
    # rmse_test = mse_score ** 0.5  #均方根误差
    # ame_score = metrics.mean_absolute_error(y_test, y_predict)# 平方绝对误差
    # r_squared_score = metrics.r2_score(y_test, y_predict)# R Squared
    #
    # print("\n_quantile_value: %f , relative_score: %f %f %f %f" % (_tmp_quantile_, mse_score,rmse_test,ame_score,r_squared_score))


    #
    # 在聚类范围[k_nim,k_max]内，计算silhouette_score
    # -----------------------------------------------------
    optimised_array = vec_line[:topN]
    _X_optimised = numpy.array(optimised_array).reshape(topN, 2)
    for i in range(2, 45):
        kmeans_model = KMeans(n_clusters=i, random_state=1).fit(_X_optimised)
        labels = kmeans_model.labels_
        s_score = metrics.silhouette_score(_X_optimised, labels, metric='euclidean')
        c_score = metrics.calinski_harabaz_score(_X_optimised, labels)
        # print("number of estimated clusters : %d, Score: %f, %f" % (i, s_score, c_score))
        print(topN, i, s_score)



    ##绘图
    # plt.figure(1)
    # plt.clf()
    #
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     ##根据lables中的值是否等于k，重新组成一个True、False的数组
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    #     plt.plot(_X_orginal[my_members, 0], _X_orginal[my_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()





    


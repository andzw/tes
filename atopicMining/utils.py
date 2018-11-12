import os
import json
def store_partitioned_docs(partitioned_docs, path="intermediate_output/partitioneddocs.txt"):
    f = open(path, 'w')
    for document in partitioned_docs:
        f.write(", ".join(" ".join(str(word) for word in phrase) for phrase in document))
        f.write("\n")
    f.flush()
    f.close()

def load_partitioned_docs(path="intermediate_output/partitioneddocs.txt"):
    f = open(path, 'r')
    partitioned_docs = []
    document_index = 0
    for line in f:
        line = line.strip()
        if len(line) < 1:
            continue
        phrases = line.split(", ")
        partitioned_doc = []
        for phrase in phrases:
            phrase_of_words = list(map(int,phrase.split(" ")))
            partitioned_doc.append(phrase_of_words)
        partitioned_docs.append(partitioned_doc)
    return partitioned_docs

def store_vocab(index_vocab, path="intermediate_output/vocab.txt"):
    """
    Stores vocabulary into a file.
    """
    f = open(path, 'w')
    for word in index_vocab:
        f.write(word+"\n")
    f.close()

def load_vocab(path="intermediate_output/vocab.txt"):
    """
    Loads vocabulary from a file.
    """
    f = open(path, 'r')
    index_vocab = []
    index = 0
    for line in f:
        index_vocab.append(line.replace("\n", ""))
    return index_vocab

def store_frequent_phrases(frequent_phrases, path='output/frequent_phrases.txt'):
    f = open(path, 'w',encoding="utf-8")
    for phrase, val in enumerate(frequent_phrases):
        f.write(str.format("{0}\t{1}\n",phrase, val[0]))
    f.flush()
    f.close()
def store_phrases(phraseDics,path):
    f = open(path, 'w', encoding="utf-8")
    for phrase in phraseDics:
        f.write(str.format("{0}\n", phrase))
    f.flush()
    f.close()
def load__phrases(path):
    result = []
    f = open(path,"r")
    for line in f:
        line = line[:-1]
        if len(line)>0:
            result.append(line)
    f.close()
    return result
def store_phrase2topic(phrase2topic,path):
    f = open (path,'w')
    for key,value in phrase2topic.items():
        f.write("{}\t{}\n".format(key,value))
    f.flush()
    f.close()
def load_phrase2topic(path):
    result={}
    f= open(path,"r")
    for line in f:
        line=line[:-1]
        temp = line.split("\t")
        if len(temp) > 1:
            result[temp[0]] = temp[1]
    f.close()
    return result
def store_config(config,path):
    with open(path,'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
def load_config(path):
    with open(path) as f:
        params = json.load(f)
    return params


def store_phrase_topics(document_phrase_topics, path="intermediate_output/phrase_topics.txt"):
    """
    Stores topic for each phrase in the document.
    """
    f = open(path, 'w')
    for document in document_phrase_topics:
        f.write(",".join(str(phrase) for phrase in document))
        f.write("\n")
    f.flush()
    f.close()

def savefile(path="intermediate_output/phrase_topics.txt"):
    """
    Stores topic for each phrase in the document.
    """
    f = open(path, 'w')

    f.write("rtestset. tsetset ,")

    f.close()


def store_most_frequent_topics(most_frequent_topics, prefix_path="output/topic"):
    file_name = prefix_path
    f = open(file_name, 'w')
    result= {}
    index = 0
    for topic_index, topic in enumerate(most_frequent_topics):
        top_temp = {}
        num = 0
        for phrase, val in topic:
            if num > 4:break
            top_temp[phrase] = val
            num +=1
        result[index] = top_temp
        index +=1
    json.dump(result, f, sort_keys=True, indent=4)
    f.flush()
    f.close()
def load_frequent_topics(path):
    result = {}
    with open(path, "r") as f:
        result = json.load(f)
    return result
def _get_string_phrase(phrase, index_vocab):
    """
    Returns the string representation of the phrase.
    """
    res = ""
    for vocab_id in phrase.split():
        if res == "":
            res += index_vocab[int(vocab_id)]
        else:
            res += " " + index_vocab[int(vocab_id)]
    return res
if __name__ == "__main__":
    path = "ModelRepertory/topic_test/topic.json"
    with open(path,"r") as f:
        result = json.load(f)
    ree = {}
    for ff in result:
        ree[ff] = result[ff].items()[:10]
    with open(path,"w") as ww:
        json.dump(result,ww,sort_keys=True, indent=4)




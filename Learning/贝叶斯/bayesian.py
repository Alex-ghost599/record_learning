#Liangyz
#2024/6/6  13:18

"""
两个问题需要处理:
1. 因为总体概率是特征概率的乘积，所以当某一个特征概率为0时，总体概率也为0.
2. 概率都是0-1之间的数，当特征很多时, 总体概率会趋近于0.
"""

import numpy as np
import re
import random


def textParse(input_str) -> list:
    listofTokens = re.split(r'\W+', input_str)
    return [tok.lower() for tok in listofTokens if len(listofTokens)>2]


def createvocablist(doclist):
    vocabset = set([])
    for doc in doclist:
        vocabset = vocabset | set(doc)
    return list(vocabset)


def setofwords2vec(vocablist, inputset):# 生成测试集的词向量,如果词在词典中，那么对应位置为1，否则为0
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!'%word)
    return returnvec


def trainNB(trainmatrix, traincategory):
    numTrainDocs = len(trainmatrix)
    numWords = len(trainmatrix[0])
    p1 = sum(traincategory)/float(numTrainDocs)
    #初始化词频
    p0num = np.ones(numWords) # 开头问题1,如果用0初始化，那么某一个特征概率为0时，总体概率也为0
    p1num = np.ones(numWords) # 拉普拉斯平滑

    p0denom = len(np.unique(traincategory))
    p1denom = len(np.unique(traincategory))

    for i in range(numTrainDocs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]# 在垃圾邮件里每一个词出现的次数
            p1denom += sum(trainmatrix[i]) # 在垃圾邮件里所有词出现的次数
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1Vec = np.log(p1num/p1denom) # 开头问题2, 为了解决概率趋近于0的问题，所以用log映射
    p0Vec = np.log(p0num/p0denom) # p1vec和p0vec 是对应词在垃圾邮件和非垃圾邮件中的概率P(wi|h+)和P(wi|h-) list形式

    return p0Vec, p1Vec, p1


def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    # log(P(w|h+)P(h+)) = sum(log(P(wi|h+))) + log(P(h+)) 通过词频*wordVec来筛选出当前邮件出现的词的词频
    p1 = np.log(p1_class) + sum(wordVec*p1Vec)
    p0 = np.log(1-p1_class) + sum(wordVec*p0Vec)# 同理 log(P(w|h-)P(h-)) = sum(log(P(wi|h-))) + log(P(h-))
    if p1 > p0:
        return 1
    else:
        return 0


def spam():
    doclicst = []
    classlist = []
    for i in range(1,26):
        wordlist = textParse(open(r'E:\Project\Learning\贝叶斯\email\spam\%d.txt'%i, 'r').read())
        doclicst.append(wordlist)
        classlist.append(1) # spam

        wordlist=textParse(open(r'E:\Project\Learning\贝叶斯\email\ham\%d.txt'%i, 'r').read())
        doclicst.append(wordlist)
        classlist.append(0)  # ham

    vocablist = createvocablist(doclicst)

    traningset = list(range(50))
    testset = []
    for i in range(10):
        randindex = int(random.uniform(0,len(traningset)))
        testset.append(traningset[randindex])
        del(traningset[randindex])

    trainmat = []
    trainclass = []
    for docindex in traningset:
        trainmat.append(setofwords2vec(vocablist, doclicst[docindex]))# 生成测试集的词向量,如果词在词典中，那么对应位置为1，否则为0
        trainclass.append(classlist[docindex])

    p0Vec, p1Vec, p1 = trainNB(np.array(trainmat), np.array(trainclass))

    errorCount = 0
    for docindex in testset:
        wordVec = setofwords2vec(vocablist,doclicst[docindex])# 生成测试集的词向量,如果词在词典中，那么对应位置为1，否则为0
        if classifyNB(np.array(wordVec), p0Vec, p1Vec, p1) != classlist[docindex]:
            errorCount += 1

    # print('the error count is', float(errorCount))
    return float(errorCount)


def main() -> None:

    for i in range(100):
        result = spam()
        if result != 0:
            print('在第{}轮测试里错了{}个'.format(i,result))


if __name__ == '__main__':
    main()







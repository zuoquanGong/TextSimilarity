# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:48:48 2019

@author: zuoquan gong
"""
import math
import time
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np

# =============================================================================
# 1.程序参数
# =============================================================================
filename='sick_label.csv'
out_filename='sick_test.out'
#test=True
test=False
# =============================================================================
# 2.数据结构
# =============================================================================
class STSstruct:#句子语义相似性测量（Semantic Text Similarity）数据结构
    def __init__(self,label,sent1,sent2):
        self.label=label#相似指标
        self.sent1=sent1#目标句A
        self.sent2=sent2#目标句B
    def set_sent1(self,sent1):#句A重置
        self.sent1=sent1
    def set_sent2(self,sent2):#句B重置
        self.sent2=sent2

# =============================================================================
# 3.函数方法
# =============================================================================
# =============================================================================
#  3.1.SICK数据集加载器
# =============================================================================
def SICK_loader(filename):
    word_dict={}#词表-word2id-dict
    word_list=[]#词表-id2word-list
    sent_dict={}#句表-sent2id-dict
    sent_list=[]#句表-id2sent-list
    
    sts_examples=[]#句子对列表-list
    
    all_sents=[]
    with open(filename,'r',encoding='utf-8') as fin:
        for i,line in enumerate(fin.readlines()):
            if i==0:
                continue
            if i>5 and test:#########
                break#########
            
            line=line.strip()
            if line=='':
                break
            sts_elems=line.split(',')
            assert(len(sts_elems)==3)
            label=sts_elems[0]
            sent1=sts_elems[1]
            sent2=sts_elems[2]
            
            ###建立句子对
            sts=STSstruct(label,sent1,sent2)
            sts_examples.append(sts)
            
            ###建立句子列表（未去重）
            all_sents.append(sent1)
            all_sents.append(sent2)
            
    ### 1.建立句子列表（去重）
    sent_list=[item[0] for item in Counter(all_sents).most_common()]#获取（句子：频度）列表，频度降序
    ### 2.建立句子词典
    for i,sent in enumerate(sent_list):
        sent_dict[sent]=i
    
    
    for sent in sent_list:
        word_list=sent.split()
        word_list[0]=word_list[0].lower()
        for word in word_list:
            if word_dict.__contains__(word):
                word_dict[word]+=1
            else:
                word_dict[word]=1
    ### 3.建立词列表
    word_list=[item[0] for item in sorted(word_dict.items(),key=lambda x:x[1],reverse=True)]
    ### 4.建立词典 word_dict
    for i,word in enumerate(word_list):
        word_dict[word]=i
    return word_dict,word_list,sent_dict,sent_list,sts_examples

# =============================================================================
#  3.2.获取tf_idf，存储在text_num*word_num的稀疏矩阵中
# =============================================================================
#本函数可重用，需要提供 1.词表-word_dict， 2.句子集合-sent_list
def tf_idf(word_dict,sent_list):
    sent_num=len(sent_list)
    word_idf={}
    
    index=0
    value=[]
    indice_col=[]
    indptr=[0]
    
    for sent in sent_list:
        sent=sent.split()
        sent[0]=sent[0].lower()
        sent_length=len(sent)
        for word_item in Counter(sent).most_common():
            value.append(word_item[1]/sent_length)
            indice_col.append(word_dict[word_item[0]])
            index+=1
        for word in set(sent):
            col_idx=word_dict[word]
            if col_idx in word_idf:
                word_idf[col_idx]+=1
            else:
                word_idf[col_idx]=1
        indptr.append(index)
    for i,v in enumerate(value):
        value[i]=v*math.log10(sent_num/word_idf[indice_col[i]])
    csr=csr_matrix((value, indice_col, indptr), shape=(len(sent_list), len(word_list)))
    
    return csr

# =============================================================================
#  3.3.计算两个向量的余弦相似度
# =============================================================================
def cosine_similarity(vec1,vec2):
    numerator=np.matmul(vec1,vec2.T)
    denominator=np.linalg.norm(vec1-vec2)
    return numerator/denominator
# =============================================================================
# 4.测试主函数
# =============================================================================
if __name__=='__main__':
    start=time.time()
    
    word_dict,word_list,sent_dict,sent_list,sts_examples=SICK_loader(filename)
    
    csr=tf_idf(word_dict,sent_list)
    print('@Test Example:')
    print(sent_list[0])
    print(word_list[:50])
    print(csr[0])

    
    for i,sts in enumerate(sts_examples):
        if i>50:
            break
        print(sts.sent1)
        print(sts.sent2)
        vec1=csr[sent_dict[sts.sent1]].toarray()
        vec2=csr[sent_dict[sts.sent2]].toarray()
        print(cosine_similarity(vec1,vec2))
        print(sts.label)
        print()
        
    #保存测试结果
    with open(out_filename,'w') as fout:
        for i,sts in enumerate(sts_examples):
            fout.write(sts.sent1)
            fout.write('\n')
            fout.write(sts.sent2)
            fout.write('\n')
            vec1=csr[sent_dict[sts.sent1]].toarray()
            vec2=csr[sent_dict[sts.sent2]].toarray()
            fout.write(str(cosine_similarity(vec1,vec2)[0][0]))
            fout.write('\t')
            fout.write(sts.label)
            fout.write('\n')
            fout.write('\n')
    end=time.time()
    print('\ntotal time:',end-start,'s')





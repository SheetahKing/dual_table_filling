import numpy as np
import spacy
import torch
from stanfordcorenlp import StanfordCoreNLP


def adj(batch_texts, max_length=None):
    # 1.生成依赖解析corenlp
    depend_list = []
    path = '/home/nlp306/Data/User_file/wba/tools/coreNLP/stanford-corenlp-4.5.3'  # coreNLP的语言模型路径:本地版
    nlp = StanfordCoreNLP(path)
    for text in batch_texts:
        # tokens = nlp.word_tokenize(text)
        depend = nlp.dependency_parse(text)
        depend_list.append(depend)
    # 2.获得adj
    dep_matrixs = []
    for i in range(len(depend_list)):
        dep_matrix = torch.zeros((max_length, max_length))
        for dep in depend_list[i]:
            i = dep[1]
            j = dep[2]
            dep_matrix[i][j] = 1
        boolean_A = dep_matrix.T > dep_matrix
        dep_matrix = dep_matrix + torch.multiply(dep_matrix.T, boolean_A) - torch.multiply(dep_matrix, boolean_A)
        dep_matrixs.append(dep_matrix.unsqueeze(0))

    dep_matrixs = torch.cat(dep_matrixs, dim=0)

    return dep_matrixs



def adj_spacy(batch_texts, max_length=None):
    # 1.加载模型
    nlp = spacy.load("en_core_web_md")

    # 2.生成dep
    list_dep = []
    for i in range(len(batch_texts)):
        doc = nlp(batch_texts[i])
        list_dep.append([])
        for token in doc:
            list_dep[i].append(token.dep_)
    print(list_dep)

class ADJ():
    def __init__(self):
        self.path = '/home/nlp306/Data/User_file/wba/tools/coreNLP/stanford-corenlp-4.5.3'  # coreNLP的语言模型路径
        self.nlp = StanfordCoreNLP(self.path)
    def adj(self, batch_texts, max_length=None):
        # 1.生成依赖解析corenlp
        depend_list = []
        for text in batch_texts:
            # tokens = nlp.word_tokenize(text)
            depend = self.nlp.dependency_parse(text)
            depend_list.append(depend)
        # 2.获得adj
        dep_matrixs = []
        for i in range(len(depend_list)):
            dep_matrix = torch.zeros((max_length, max_length))
            for dep in depend_list[i]:
                i = dep[1]
                j = dep[2]
                dep_matrix[i][j] = 1
            boolean_A = dep_matrix.T > dep_matrix
            dep_matrix = dep_matrix + torch.multiply(dep_matrix.T, boolean_A) - torch.multiply(dep_matrix, boolean_A)
            dep_matrixs.append(dep_matrix.unsqueeze(0))

        dep_matrixs = torch.cat(dep_matrixs, dim=0)

        dep_matrixs_tensor = torch.tensor(dep_matrixs, dtype=torch.float32, device='cuda')

        return dep_matrixs_tensor

# tests = ['I charge it at night', 'He charge it at night', '"She charge it at night']
# l = 10

# cl = ADJ()
# a = cl.adj(tests, l)
# b = cl.adj(tests, 11)
# c = cl.adj(tests, 12)
# print(a.shape)
# print(b.shape)
# print(c.shape)
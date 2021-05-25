from src.sample.Domain import Domain
from src.sample.Clust import Clust
from tqdm import tqdm
import spacy
spacy.prefer_gpu()
from spacy.lang.en import English
import json
import os


def makeDomain(DomainName, clusts_path):
    clustList = []
    with open(clusts_path, 'r') as fin:
        all_clusts = json.load(fin)
        for i, c in enumerate(all_clusts):
            clust = Clust('clust%d' % (i), c, DomainName)
            clustList.append(clust)

    return Domain(DomainName, clustList)


def selectTextForClust(clust, ref_path):
    with open(ref_path, "r", encoding='utf-8') as ref_r:
        lines = ref_r.readlines()
        txtClustNum = 0
        for line in lines:
            titleAndtitleText = line.split("\t\t")
            title = titleAndtitleText[0]

            if title in clust.getTittleList():
                titleText = titleAndtitleText[1]
                txtClustNum += clust.judge(title, titleText)
            # print("#"*20)

        return txtClustNum


def work(args):
    nlp_spacy = English()
    nlp_spacy.add_pipe(nlp_spacy.create_pipe('sentencizer'))
    modes = ["train", "valid", "test"]
    allDomain = ["animal", "company", "film"]
    for mode in modes:
        allParaCount = 0
        for domainName in allDomain:
            clusts_path = os.path.join(args.classifier_dir, domainName, 'TopicList.txt')
            domain = makeDomain(domainName, clusts_path)
            path_titleTxt = os.path.join(args.classifier_dir, domainName, '%s.TitleText.txt' % (mode))
            fout = open(path_titleTxt, 'w')

            for clust in domain.getClustList():
                ref_path = os.path.join(args.data_dir, clust.getDomainName(), '%s.TitleText.txt' % (mode))
                allParaCount += selectTextForClust(clust, ref_path)

            for clust in tqdm(domain.getClustList()):
                clust.writeTxt(fout, nlp_spacy)

            fout.close()
        print(mode, allParaCount)


if __name__ == '__main__':
    pass

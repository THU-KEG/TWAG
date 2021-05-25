#coding=utf-8
import re
from src.sample.CoutainPictureError import CoutainPictureError

class Clust():
    def __init__(self, name, titleList, domainName):
        self.name = name
        self.titleList = titleList
        self.domainName = domainName
        self.title2texts = {}

    def __repr__(self):
        return self.name

    def getName(self):
        return self.name

    def getDomainName(self):
        return self.domainName

    def getTittleList(self):
        return self.titleList

    def judge(self, titleName, titleText):
        selectedTextList = []
        rawTextList = titleText.split("::;")
        for para in rawTextList:
            searchObj = re.search(r"[a-zA-Z]+://[^\s]*[.com|.cn]", para)
            if searchObj:
                continue
            else:
                searchObj = re.search(r"\#|\+|\^|\<|\>|\@|=|\-|\&|\*", para)
                if searchObj:
                    continue

                if len(para) <= 1:
                    continue
                
                try:
                    para = self.parse(para)
                except CoutainPictureError:
                    continue
                selectedTextList.append(para)
        
        self.title2texts[titleName] = selectedTextList
        return len(selectedTextList)

    def parse(self, para):
        def parseInnerStr(innerStr):
            pattern2 = re.compile(r'.+?\|(.+)')
            searchObj2 = pattern2.search(innerStr)
            if searchObj2:
                trueName = searchObj2.group(1)
                if re.search("\|", trueName):
                    raise CoutainPictureError
                return trueName
            else:
                return innerStr

        pattern1 = re.compile(r'\[\[(.+?)\]\]')
        searchObj = pattern1.search(para)
        while searchObj:
            innerStr = searchObj.group(1)
            start = searchObj.span()[0]
            end = searchObj.span()[1]

            trueName1 = parseInnerStr(innerStr)
            para = para[:start] + trueName1 + para[end:]
            searchObj = pattern1.search(para)
        return para

    def writeTxt(self, txtFile, nlp_spacy):

        for titleName, selectedTextList in self.title2texts.items():
            for para in selectedTextList:
                document = nlp_spacy(para)
                sentenceList = list(document.sents)
                for sentence in sentenceList:
                    raw_sentence = str(sentence)

                    raw_sentence = raw_sentence.strip()
                    if (raw_sentence == ''):
                        continue

                    line = raw_sentence + "\t\t" + titleName + "\t\t" + self.name + "\n"
                    txtFile.write(line)


if __name__ == '__main__':
    try:
        raise CoutainPictureError
    except CoutainPictureError:
        print("has pic")

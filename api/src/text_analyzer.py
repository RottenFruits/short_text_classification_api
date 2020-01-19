import MeCab
import numpy as np
import pandas as pd

class TextAnalyzer:        
    def raw_morphological_analize(self, text):
        m = MeCab.Tagger('-Ochasen')
        return(m.parse(text))
    
    def morphological_analize(self, text, target_pos = None):
        #テキストを形態素解析し指定した品詞のみ取り出して返す
        #hyouso = np.array([])
        genkei = np.array([])
        hinshi = np.array([])
        target_pos_pattern = self.pos_pattern_create(target_pos)
        
        words_tmp = self.raw_morphological_analize(text)
        words = words_tmp.split("\n")
        for word in words:
            if word.split("\t")[0] == "EOS":
                break
            #print(word.split("\t"))
            tags = word.split("\t")
            if target_pos_pattern.search(tags[3]) != None: #指定した品詞のみ取り出す
                #hyouso.append(tags[0]) #表層形
                #tags[1] #ヨミ
                genkei = np.append(genkei, tags[2]) #原型
                hinshi = np.append(hinshi, tags[3].split("-")[0]) #品詞
                #tags[4] #活用形
                #tags[5] #活用型
                #print(len(w.split("\t")))            
        return(np.array([genkei, hinshi]))
    
    def pos_pattern_create(self, pos):
        #品詞絞る用の正規表現パターン生成
        import re
        if pos:
            i = 0
            for p in pos:
                pos[i] = "^" + p
                i += 1
            pos = "|".join(pos)
        else:
            pos = ".*"
        pos_pattern = re.compile(pos)
        return(pos_pattern)
    
    def base_bi_gram_create(self, arr):
        #単純にバイグラムを作る
        i = 0
        bi_gram = np.empty((0, 2), int)
        while i < (arr.size - 1):
            bi_gram = np.append(bi_gram, np.array([[arr[i], arr[i+1]]]), axis = 0)
            i += 1
        return(bi_gram)
    
    def bi_gram_pos_order_filter(self, genkei, hinshi, target_pos_order):
        #指定した品詞順のバイグラムパターンを取り出して返す 
        if genkei.size == 0:
            print("array is null")
            return
        bi_gram_genkei = np.empty((0, 2), int)
        i = 0
        while i < (genkei.size - 1):
            if hinshi[i] == target_pos_order[0] and hinshi[i+1] == target_pos_order[1]:
                bi_gram_genkei = np.append(bi_gram_genkei, np.array([[genkei[i], genkei[i+1]]]), axis = 0)
                g = np.array([])
            i += 1
        return(bi_gram_genkei)
    
    def bi_gram_pos_order_filter2(self, genkei, hinshi, target_pos_order):
        #指定した品詞順のバイグラムパターンを取り出して返す 
        #同じ品詞が連続している場合結合する
        if genkei.size == 0:
            print("array is null")
            return
        bi_gram_genkei = np.empty((0, 2), int)
        g = np.array([])
        i = 0
        while i < (genkei.size - 1):
            if hinshi[i] == target_pos_order[0] and hinshi[i+1] == target_pos_order[0]:
                if g.size == 0:
                    g = genkei[i]
                g = np.append(g, genkei[i+1])
            if hinshi[i] == target_pos_order[0] and hinshi[i+1] == target_pos_order[1]:
                if g.size == 0:
                    g = genkei[i]
                if g.size >= 2:
                    g = "-".join(g)
                bi_gram_genkei = np.append(bi_gram_genkei, np.array([[g, genkei[i+1]]]), axis = 0)
                g = np.array([])
            i += 1
        return(bi_gram_genkei)
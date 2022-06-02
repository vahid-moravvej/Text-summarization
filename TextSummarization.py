from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import pickle
import os
import itertools
import random
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import Linear,ReLU,Sigmoid, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d

Sentence_Docs={}
Sentence_Docs_Train={}
Sentence_Docs_Test={}
Sentence_Docs_Pre={}
Abstract={}
Abstract_Train={}
Abstract_Test={}
Feature={}
Feature_Train={}
Feature_Test={}
Rouge_Sentence={}
Number_Doc=0
Num_Sentence_Doc=0
Rouge_Worst={}
Rouge_Best={}
Languege=''
###### Parameters ########
Size_Noise=10
Learning_Rate=0.15
Number_Epoch=3
Batch_Size=30
Num_Features=13
Num_Sentence_Doc=50
W1=0.5
W2=1
Best_Lost=0

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
def save_data(filename,data):
    with open(filename, 'wb') as f:
        pickle.dump(data,f)
class Generator(Module):   
    def __init__(self,size_noise):
        super(Generator, self).__init__()
        self.size_noise=size_noise
        self.cnn_layers = Sequential(
            BatchNorm2d(1),
            Conv2d(1, 8, kernel_size=(3,2),padding=(1,1) ,stride=(1,1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2,1), stride=(2,2)),
            Conv2d(8, 16, kernel_size=(3,2), stride=(2,2), padding=(1,1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(112+self.size_noise,60),
            #Tanh(),
            ReLU(inplace=True),
            Linear(60,Num_Sentence_Doc),
            Sigmoid()
        ) 
    def forward(self, x,noise):    
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,noise), -1)
        x = self.linear_layers(x)
        return x
class Discriminator(Module):   
    def __init__(self):
        super(Discriminator, self).__init__() 
        self.cnn_layers = Sequential(
            BatchNorm2d(1),
            Conv2d(1, 8, kernel_size=(3,2),padding=(1,1) ,stride=(1,1)), 
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2,1), stride=(2,2)),
            Conv2d(8, 4, kernel_size=(3,2), stride=(2,2), padding=(1,1)),
            ReLU(inplace=True),
            #Tanh(),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(28+Num_Sentence_Doc, 15),
            ReLU(inplace=True),
            #Tanh(),
            Linear(15, 20),
            ReLU(inplace=True),
            #Tanh(),
            Linear(20, 1),
            Sigmoid()
        )       
    def forward(self, x,sentence_prob):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,sentence_prob), -1)
        x = self.linear_layers(x)
        return x
def get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set
def split_into_words(sentences):
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))
def get_word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0
    words = split_into_words(sentences)
    return get_ngrams(n, words)
def Rouge_N(evaluated_sentences, reference_sentences, n=2):
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    evaluated_ngrams = get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count
    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall+ 1e-8))
    #l=rouge_l(reference_sentences[0],evaluated_sentences[0])
    return {"f": f1_score, "p": precision, "r": recall}
def Commput_Rouge_Sentence(listsentence,abstract):
    global Num_Sentence_Doc
    temp=[]
    bigger_threshold=[]
    i=0
    for sen in listsentence:
        if(i>=Num_Sentence_Doc):
           break
        t=W2*Rouge_N([sen],[abstract],2)['p']+W1*Rouge_N([sen],[abstract],1)['p']
        temp.append(t)
        i=i+1
    avg=sum(temp)/len(temp)
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    bigger_threshold=[]
    for i in range(len(temp)):
        if(temp[i]>avg):
            bigger_threshold.append(1) 
        else:
            bigger_threshold.append(0) 
    return bigger_threshold
def Evaluation(generator,discriminator,e):
    global Best_Lost,Languege
    values=[]
    for key,value in Sentence_Docs_Train.items():
        values.append(Ret_Feature(key))
    input=torch.tensor(values)
    input=input.view(len(Sentence_Docs_Train),-1,Num_Sentence_Doc)
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (len(Sentence_Docs_Train), Size_Noise))))
    output_g=generator.forward(input.unsqueeze(1).float(),z)
    i=0
    avg_rouge_f2=0
    avg_rouge_p2=0
    avg_rouge_r2=0
    avg_rouge_f1=0
    avg_rouge_p1=0
    avg_rouge_r1=0
    for  key,value in Sentence_Docs_Train.items():
        bigger_threshold=[]
        avg=sum(output_g[i])/len(output_g[i])
        for k in range(Num_Sentence_Doc):
          if(output_g[i][k]>avg):
              bigger_threshold.append(1)
          else:
              bigger_threshold.append(0)  
        sentences=""
        for j  in range(len(bigger_threshold)):
            if(bigger_threshold[j]==1):
                if(j<len(Sentence_Docs_Train[key])):
                    sentences=sentences+Sentence_Docs_Train[key][j]
        rouge_2=Rouge_N([sentences],[Abstract_Train[key]],2)
        avg_rouge_p2=avg_rouge_p2+rouge_2['p']
        avg_rouge_f2=avg_rouge_f2+rouge_2['f']
        avg_rouge_r2=avg_rouge_r2+rouge_2['r']
        rouge_1=Rouge_N([sentences],[Abstract_Train[key]],1)
        avg_rouge_p1=avg_rouge_p1+rouge_1['p']
        avg_rouge_f1=avg_rouge_f1+rouge_1['f']
        avg_rouge_r1=avg_rouge_r1+rouge_1['r']
        i=i+1  
    avg_rouge_p2=avg_rouge_p2/len(Sentence_Docs_Train)
    avg_rouge_f2=avg_rouge_f2/len(Sentence_Docs_Train)
    avg_rouge_r2=avg_rouge_r2/len(Sentence_Docs_Train)
    avg_rouge_p1=avg_rouge_p1/len(Sentence_Docs_Train)
    avg_rouge_f1=avg_rouge_f1/len(Sentence_Docs_Train)
    avg_rouge_r1=avg_rouge_r1/len(Sentence_Docs_Train)
    x_p2=0
    x_r2=0
    x_p1=0
    x_r1=0
    if(avg_rouge_p2>=Best_Lost):
       Best_Lost=avg_rouge_p2
       if not os.path.exists('Summarization/' + Languege + '/Model'):
           os.makedirs('Summarization/' +Languege + '/Model')
       torch.save(generator.state_dict(),'Summarization/' +Languege + '/Model/model.model')
    print("Average Rouge For Train(p_2): ",avg_rouge_p2," best rouge: ",Best_Lost)
    print("Average Rouge For Train(f_2): ",avg_rouge_f2) 
    print("Average Rouge For Train(r_2): ",avg_rouge_r2)  
    print("Average Rouge For Train(p_1): ",avg_rouge_p1)  
    print("Average Rouge For Train(f_1): ",avg_rouge_f1)  
    print("Average Rouge For Train(r_1): ",avg_rouge_r1)
    #######Test
    values=[]
    for key,value in Sentence_Docs_Test.items():
        values.append(Ret_Feature(key))
    input=torch.tensor(values)
    input=input.view(len(Sentence_Docs_Test),-1,Num_Sentence_Doc)
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (len(Sentence_Docs_Test), Size_Noise))))
    output_g=generator.forward(input.unsqueeze(1).float(),z)
    i=0
    avg_rouge_f2=0
    avg_rouge_p2=0
    avg_rouge_r2=0
    avg_rouge_f1=0
    avg_rouge_p1=0
    avg_rouge_r1=0
    for  key,value in Sentence_Docs_Test.items():
        bigger_threshold=[]
        avg=sum(output_g[i])/len(output_g[i])
        for k in range(Num_Sentence_Doc):
          if(output_g[i][k]>avg):
              bigger_threshold.append(1)
          else:
              bigger_threshold.append(0) 
        sentences=""  
        for j  in range(len(bigger_threshold)):
            if(bigger_threshold[j]==1):
                if(j<len(Sentence_Docs_Test[key])):
                    sentences=sentences+Sentence_Docs_Test[key][j]
        rouge_2=Rouge_N([sentences],[Abstract_Test[key]],2)
        avg_rouge_p2=avg_rouge_p2+rouge_2['p']
        avg_rouge_f2=avg_rouge_f2+rouge_2['f']
        avg_rouge_r2=avg_rouge_r2+rouge_2['r']
        rouge_1=Rouge_N([sentences],[Abstract_Test[key]],1)
        avg_rouge_p1=avg_rouge_p1+rouge_1['p']
        avg_rouge_f1=avg_rouge_f1+rouge_1['f']
        avg_rouge_r1=avg_rouge_r1+rouge_1['r']  
        i=i+1
    avg_rouge_p2=avg_rouge_p2/len(Sentence_Docs_Test)+x_p2
    avg_rouge_f2=avg_rouge_f2/len(Sentence_Docs_Test)
    avg_rouge_r2=avg_rouge_r2/len(Sentence_Docs_Test)+x_r2
    avg_rouge_p1=avg_rouge_p1/len(Sentence_Docs_Test)+x_p1
    avg_rouge_f1=avg_rouge_f1/len(Sentence_Docs_Test)
    avg_rouge_r1=avg_rouge_r1/len(Sentence_Docs_Test)+x_r1
    print("Average Rouge For Test(p_2): ",avg_rouge_p2)  
    print("Average Rouge For Test(f_2): ",avg_rouge_f2) 
    print("Average Rouge For Test(r_2): ",avg_rouge_r2)  
    print("Average Rouge For Test(p_1): ",avg_rouge_p1)  
    print("Average Rouge For Test(f_1): ",avg_rouge_f1)  
    print("Average Rouge For Test(r_1): ",avg_rouge_r1)
def Ret_Feature(doc_index):
    features=Feature[doc_index]
    features_sen=[]
    for i in range(Num_Sentence_Doc): 
        if(i<len(features)):
            fea=features[i]
        else:
            fea=[0 for i in range(Num_Features)]
        features_sen.append(fea)     
    return (features_sen)
def Make_Data(keys,Rouge_List,count,isbest):
    keys_d=[]
    values_d=[]
    input_d=[]
    rouge_sen=[]   
    le=0
    for value in keys:
        if(count==0):
            len_list=len(Rouge_List[value])
        else:
            len_list=count
        sort_rouge=[]
        for j in range(len(Rouge_List[value])):
            sort_rouge.append(Calclute_Rouge_List(Rouge_List[value][j],value))
        if(isbest):
            sort_rouge=[j[0] for j in sorted(enumerate(sort_rouge), key=lambda x:x[1])]                   
            le=le+len_list
            for j in range(len(sort_rouge)-1,len(sort_rouge)-(len_list+1),-1):
                keys_d.append(value)
                values_d.append(Ret_Feature(value))
                rouge_sen.append(Rouge_List[value][sort_rouge[j]])
        else:
            sort_rouge=[j[0] for j in sorted(enumerate(sort_rouge), key=lambda x:x[1])]                   
            le=le+len_list
            for j in range(0,len_list):
                keys_d.append(value)
                values_d.append(Ret_Feature(value))
                rouge_sen.append(Rouge_List[value][sort_rouge[j]])
    input_d=torch.tensor(values_d)
    input_d=input_d.view(le,-1,Num_Sentence_Doc)
    rouge_sen=torch.tensor(rouge_sen)
    rouge_sen=rouge_sen.float()
    return le,rouge_sen,input_d
def Training_Model(generator, discriminator, optimizer_G, optimizer_D):
    global Batch_Size,Num_Sentence_Doc,Sentence_Docs_Train,Number_Epoch,Rouge_Sentence_Train,Loss_Epoch
    batch_size=Batch_Size
    keys=[]
    values=[]
    adversarial_loss = torch.nn.MSELoss()
    best_loss_g=100000000
    batch_best_g=0
    for i in range(Number_Epoch):
        shuffle_doc=list(Sentence_Docs_Train)
        random.shuffle(shuffle_doc)
        batch=0
        loss_epoch_g=0    
        loss_epoch_d=0
        for k in range((len(Sentence_Docs_Train)//batch_size)+1):
            if(k*batch_size==len(Sentence_Docs_Train)):
                break
            if(k==len(Sentence_Docs_Train)//batch_size):
                sel_doc=shuffle_doc[k*batch_size:]
            else:       
                sel_doc=shuffle_doc[k*batch_size:(k+1)*batch_size]
            for value in sel_doc:     
                keys.append(value)
                values.append(Ret_Feature(value))
            Batch_Size=len(keys)
            input=torch.tensor(values)#(Batch_Size,Num_Sentence_Doc,Num_Features,)
            input=input.view(Batch_Size,-1,Num_Sentence_Doc)#(Batch_Size,Num_Features,Num_Sentence_Doc)
            batch=batch+1
            optimizer_G.zero_grad()
            valid = Variable(torch.FloatTensor(Batch_Size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(Batch_Size, 1).fill_(0.0), requires_grad=False)
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (Batch_Size, Size_Noise))))
            output_g=generator.forward(input.unsqueeze(1).float(),z)
            validity_fake = discriminator(input.unsqueeze(1).float(), output_g)
            g_loss = adversarial_loss(validity_fake, valid)
            loss_epoch_g=loss_epoch_g+g_loss
            g_loss.backward(retain_graph=True)
            optimizer_G.step()
            if(i%10==0):
                optimizer_D.zero_grad()
                le,rouge_sen,input_d=Make_Data(keys,Rouge_Best,0,True)
                validity_real1 = discriminator(input_d.unsqueeze(1).float(), rouge_sen)
                valid = Variable(torch.FloatTensor(le, 1).fill_(1.0), requires_grad=False)
                d_real_loss =adversarial_loss(validity_real1, valid)
                le,rouge_sen1,input_d=Make_Data(keys,Rouge_Worst,1,False)
                validity_fake1 = discriminator(input_d.unsqueeze(1).float(), rouge_sen1)
                fake = Variable(torch.FloatTensor(le, 1).fill_(0.0), requires_grad=False)
                d_fake_loss =adversarial_loss(validity_fake1, fake)
                fake = Variable(torch.FloatTensor(Batch_Size, 1).fill_(0.0), requires_grad=False)
                validity_fake2 = discriminator(input.unsqueeze(1).float(), output_g)
                d_fake_loss=d_fake_loss+adversarial_loss(validity_fake2, fake)
                d_loss = d_real_loss
                loss_epoch_d=loss_epoch_d+d_loss
                d_loss.backward()
                optimizer_D.step()  
            keys=[]
            values=[]
        Evaluation(generator,discriminator,i)
        if(loss_epoch_g<best_loss_g):
            best_loss_g=loss_epoch_g
            batch_best_g=i+1
        print("Epoch:",i+1," loss epoch(G):",loss_epoch_g.item()," best loss epoch(G):",batch_best_g,":",best_loss_g.item())
        print("------------------------------------------------------")
def feature_mat(sentences,sentences_pre):
    sent=sentences[:]
    feature=pd.DataFrame() 
    
    #10 frequent word
    from collections import Counter   
    temp=[]
    for i in sentences_pre:
        for j in i.split():
            temp.append(j)
    Counter = Counter(temp)        
    frequent = Counter.most_common(10)
    thematic=[]
    for i in frequent:
        thematic.append(i[0])
    temp=[]
    for i,line in enumerate(sentences_pre):
        count=0
        for j,word in enumerate(line.split()):
            if word in thematic:
                count+=1
        temp.append(count/(len(line.split())+1e-10))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se = pd.Series(temp)       
    feature['thematic'] = se.values
    
    #sentence pos
    temp=[]
    for i in range(Num_Sentence_Doc):
        j=i+1
        val=1-((j-1)/(Num_Sentence_Doc-1))
        temp.append(val)
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            j=i+1
            val=1-((j-1)/(Num_Sentence_Doc-1))
            temp.append(val)
    se=pd.Series(temp)   
    feature['sen_pos']=se.values 
    
    #sen_length
    temp=[]
    for i in sentences_pre:
        k=(i.split().__len__())
        temp.append(k) 
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)   
    se=se/max(se)
    feature['sen_length']=se.values      
    
    #numeral
    temp=[]
    for i in sentences_pre:
        count=0
        for j in i.split():
            if j.isnumeric():
                count+=1
        temp.append(count/(len(i.split())+1e-10)) 
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['numeral']=se.values
    
    
    #named entity recog
    temp=[]
    for i in sent:
        temp.append(nltk.word_tokenize(i))
    ne=set()    
    for i in temp:    
        for chunk in nltk.ne_chunk(nltk.pos_tag(i)):
            if hasattr(chunk, 'label'):#have doubt see ho it works
               ne.add(chunk[0][0])
    ne=list(ne)           
    sne=[]
    for i in ne:
        sne.append(nltk.PorterStemmer().stem(i))
    temp=[]
    for i in sentences_pre:       
        count=0
        for j in i.split():
            if j in sne:
               count+=1
        temp.append(count/(len(i.split())+1e-10))    
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['name_entity']=se.values
    #tf_isf
    isf={}
    word=[]
    for i in sentences_pre:
        for j in i.split():
            word.append(j)
    from collections import Counter     
    Counter = Counter(word)        
    freq = Counter.most_common()
    for tup in freq:
        isf[tup[0]]=tup[1]  
    tf_isf=[]
    for i in sentences_pre:
        l=i.split().__len__()
        temp=[]
        for j in i.split():
            temp.append(j)
        from collections import Counter     
        Counter = Counter(temp)        
        freq = Counter.most_common()
        tf={}
        for i in freq:
            tf[i[0]]=i[1]
        sum=0
        for i,j in tf.items():
            sum+=tf[i]*isf[i]
        tf_isf.append(sum)
    if(len(tf_isf)>Num_Sentence_Doc):
        tf_isf=tf_isf[0:Num_Sentence_Doc]
    if(len(tf_isf)<Num_Sentence_Doc):
        for i in range(len(tf_isf),Num_Sentence_Doc):
            tf_isf.append(0) 
    se=pd.Series(tf_isf)  
    se=se/(max(se))
    feature['tf_isf']=se.values  
    
    #centroid similarity
    big=0
    for i in tf_isf:
        if big<i:
            big=i
    index=tf_isf.index(big)        
    lst=sentences_pre[index].split()       
    sent_sim=[]
    for i in sentences_pre:
        l=i.split().__len__()
        sim=0
        for j in i.split():
            if j in lst:
                sim+=1
        sent_sim.append(sim/(l+1e-10))         
    if(len(sent_sim)>Num_Sentence_Doc):
        sent_sim=sent_sim[0:Num_Sentence_Doc]
    if(len(sent_sim)<Num_Sentence_Doc):
        for i in range(len(sent_sim),Num_Sentence_Doc):
            sent_sim.append(0)    
    se=pd.Series(sent_sim)        
    feature['cent_sim']=se.values   
    
    #stop_word
    temp=[]
    stop_words=set(stopwords.words('english'))    
    for i in sentences:
        count=0
        for j in i.split():
            if j in stop_words:
                count+=1
        temp.append(1-(count/(len(i.split())+1e-10)))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['stop_word']=se.values
    
    #nouns_phrases
    from textblob import TextBlob
    temp=[]
    for i in sentences:
        blob = TextBlob(i)
        temp.append(len(blob.noun_phrases)/(len(i.split())+1e-10))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['nouns_phrases']=se.values
    
    #adjective
    from textblob import TextBlob
    temp=[]
    for i in sentences:
        is_adjective = lambda pos: pos[:2] == 'JJ' or  pos[:2]=='JJR' or pos[:2]=='JJS'
        tokenized = nltk.word_tokenize(i)
        adjective = [word for (word, pos) in nltk.pos_tag(tokenized) if is_adjective(pos)] 
        temp.append(len(adjective)/(len(i.split())+1e-10))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['adjective']=se.values
    
    #adverb
    from textblob import TextBlob
    temp=[]
    for i in sentences:
        is_adverb = lambda pos: pos[:2] == 'RB' or  pos[:2]=='RBR' or pos[:2]=='RBS'
        tokenized = nltk.word_tokenize(i)
        adverb = [word for (word, pos) in nltk.pos_tag(tokenized) if is_adverb(pos)] 
        temp.append(len(adverb)/(len(i.split())+1e-10))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['adverb']=se.values
    
    
    #verb
    from textblob import TextBlob
    temp=[]
    for i in sentences:
        is_verb = lambda pos: pos[:2] == 'VB' or  pos[:2]=='VBZ' or pos[:2]=='VBP' or pos[:2]=='VBD' or pos[:2]=='VBN' or pos[:2]=='VBG'  
        tokenized = nltk.word_tokenize(i)
        verb = [word for (word, pos) in nltk.pos_tag(tokenized) if is_verb(pos)] 
        temp.append(len(verb)/(len(i.split())+1e-10))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['verb']=se.values
    
    #noun
    from textblob import TextBlob
    temp=[]
    for i in sentences:
        is_noun = lambda pos: pos[:2] == 'NN' or  pos[:2]=='NNS' or pos[:2]=='NNP' or pos[:2]=='NNPS'  
        tokenized = nltk.word_tokenize(i)
        noun = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        temp.append(len(noun)/(len(i.split())+1e-10))
    if(len(temp)>Num_Sentence_Doc):
        temp=temp[0:Num_Sentence_Doc]
    if(len(temp)<Num_Sentence_Doc):
        for i in range(len(temp),Num_Sentence_Doc):
            temp.append(0)
    se=pd.Series(temp)        
    feature['noun']=se.values
    return(feature)
def Preprocess(string):
    sentence_clear=""
    character_stopwords=['.','_','[',']','(',')',"=","+","&","%","*","#","@","'","\"",","]
    sent=[]
    for j in sent_tokenize(string):
        sent.append(j)                   
    x=sent[:]
    stop_words=set(stopwords.words('english'))    
    temp=[]
    for lst in x:
        temp.append(word_tokenize(lst))   
    for i,lst in enumerate(temp):
        temp[i]=[w for w in lst if (not w in stop_words) and (not w in character_stopwords) ]   
    for i,lst in enumerate(temp):
        for j,word in enumerate(lst):
            temp[i][j]=nltk.PorterStemmer().stem(word)
    x=[' '.join(i) for i in temp] 
    for sen in x:
       sentence_clear=sentence_clear+sen  
    return(sentence_clear)
def ReturnLlstSentence(sentences):
        return sent_tokenize(sentences)
def Calclute_Rouge_List(rouge,key):
    sentences=""  
    for j  in range(Num_Sentence_Doc):
        if(rouge[j]==1):
            if(j<len(Sentence_Docs_Train[key])):
                sentences=sentences+Sentence_Docs_Train[key][j]
    avg_rouge=Rouge_N([sentences],[Abstract_Train[key]],2)['p'] 
    return avg_rouge
def Calclute_Best_Rouge(index,count):
    rouge=[]
    for i in range(5):
        temp= [0] * Num_Sentence_Doc
        indexs=random.sample(range(0, Num_Sentence_Doc), count)
        for j in range(count):
            temp[indexs[j]]=1
        best_rouge_list=temp.copy()
        best_rouge=Calclute_Rouge_List(temp,index)
        for k in range(6):
            ones=[]
            zeros=[]
            for n in range(len(temp)):
                if(temp[n]==1):
                    ones.append(n)
                else:
                    zeros.append(n)
            zero=random.choice(zeros)
            one=random.choice(ones)
            temp[zero]=1
            temp[one]=0
            if(Calclute_Rouge_List(temp,index)>best_rouge):
                best_rouge=Calclute_Rouge_List(temp,index)
                best_rouge_list=temp.copy()
            else:
                temp=best_rouge_list.copy()
        co=True
        for j in range(len(rouge)):
            if(rouge[j]==temp):
                co=False
                break
        if(co):
            rouge.append(temp)
        #print(best_rouge)
        #print(temp)
    return rouge
def Calclute_Worst_Rouge(index,count):
    rouge=[]
    for i in range(1):
        temp= [0] * Num_Sentence_Doc
        indexs=random.sample(range(0, Num_Sentence_Doc), count)
        for j in range(count):
            temp[indexs[j]]=1
        worst_rouge_list=temp.copy()
        worst_rouge=Calclute_Rouge_List(temp,index)
        for k in range(6):
            ones=[]
            zeros=[]
            for n in range(len(temp)):
                if(temp[n]==1):
                    ones.append(n)
                else:
                    zeros.append(n)
            zero=random.choice(zeros)
            one=random.choice(ones)
            temp[zero]=1
            temp[one]=0
            if(Calclute_Rouge_List(temp,index)<worst_rouge):
                worst_rouge=Calclute_Rouge_List(temp,index)
                worst_rouge_list=temp.copy()
            else:
                temp=worst_rouge_list.copy()
        co=True
        for j in range(len(rouge)):
            if(rouge[j]==temp):
                co=False
                break
        if(co):
            rouge.append(temp)
        #print(best_rouge)
        #print(temp)
    return rouge
def main(Address_Data,Abstract_Data,languege):
    global Num_Sentence_Doc,Sentence_Docs,Sentence_Docs_Train,Sentence_Docs_Test,Number_Doc,Rouge_Sentence,Rouge_Sentence_Train,Rouge_Sentence_Test,Abstract,Abstract_Train,Abstract_Test,Sentence_Docs_Pre,Sentence_Docs_Pre_Train,Sentence_Docs_Pre_Test,Feature,Feature_Train,Feature_Test,Rouge_Worst,Rouge_Best,Languege
    indexes=[]
    index_train={}
    index_test={}
    Languege=languege
    if not os.path.exists('Summarization/'+languege):
        os.makedirs('Summarization/'+languege)
    for filename in os.listdir(Address_Data):
         i=int(filename.replace(".txt", ""))
         address=Address_Data+'/'+filename
         f = open(address, "r", encoding="utf8")
         input_sentences=f.read()
         Sentence_Docs[i] = ReturnLlstSentence(input_sentences)
         temp=[]
         for j in range(len(Sentence_Docs[i])):
            temp.append(Preprocess(Sentence_Docs[i][j]))
         Sentence_Docs_Pre[i]=temp
         feature=feature_mat(Sentence_Docs[i],Sentence_Docs_Pre[i])
         X=feature.iloc[:,:].values
         lst=[]
         for j in range(X.shape[0]):
             n=[]
             for k in range(X.shape[1]):
                n.append(X[j,k])
             lst.append(n)
         Feature[i]=lst
         address=Abstract_Data+'/'+str(i)+".txt"
         f = open(address, "r", encoding="latin-1")
         Abstract[i]=f.read()
    Number_Doc=len(Sentence_Docs)
    for i in range(Number_Doc):
         Rouge_Sentence[i+1]=Commput_Rouge_Sentence(Sentence_Docs[i+1],Abstract[i+1])
         indexes.append(i+1)
    random.shuffle(indexes)
    for i in range(0,int(2/3*len(indexes))):
          index_train[indexes[i]]=Sentence_Docs[indexes[i]]
    for i in range(int(2/3*len(indexes)),len(indexes)):
          index_test[indexes[i]] = Sentence_Docs[indexes[i]]
    for key,value in index_train.items():
           Sentence_Docs_Train[key]=Sentence_Docs[key]
           Abstract_Train[key]=Abstract[key]
           Feature_Train[key]=Feature[key]
           Rouge_Best[key]=Calclute_Best_Rouge(key,20)
           Rouge_Worst[key]=Calclute_Worst_Rouge(key,20)
    for key,value in index_test.items():
           Sentence_Docs_Test[key]=Sentence_Docs[key]
           Abstract_Test[key]=Abstract[key]
           Feature_Test=Feature[key]
    generator = Generator(Size_Noise)
    discriminator = Discriminator()
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=Learning_Rate)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=Learning_Rate)
    Training_Model(generator, discriminator, optimizer_G, optimizer_D)
def Summarization(text,number,languege):
    summarize=""
    generator=Generator(Size_Noise)
    generator.load_state_dict(torch.load("Summarization/"+languege+"/Model/model.model"))
    sens = ReturnLlstSentence(text)
    len_sens=len(sens)
    if(number<=0):
        return ""
    if(number>=len_sens):
        return text
    pre_sens=[]
    for j in range(len(sens)):
        pre_sens.append(Preprocess(sens[j]))
    feature=feature_mat(sens,pre_sens)
    X=feature.iloc[:,:].values
    lst=[]
    for j in range(X.shape[0]):
        n=[]
        for k in range(X.shape[1]):
            n.append(X[j,k])
        lst.append(n)
    input=torch.tensor(lst)
    input=input.view(1,-1,Num_Sentence_Doc)
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, Size_Noise))))
    output_g=generator.forward(input.unsqueeze(1).float(),z)
    output_g=output_g[0][0:len_sens]
    output_g=output_g.tolist()
    indexs=np.argsort(output_g)
    indexs=indexs.tolist()
    indexs=indexs[len(indexs)-number:]
    indexs.sort()
    for i in range(number):
        summarize=summarize+sens[indexs[i]]
    return summarize






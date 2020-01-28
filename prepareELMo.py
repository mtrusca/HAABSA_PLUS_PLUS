import numpy as np
import tensorflow_hub as hub
from config import *
from utils import load_w2v

url = "https://tfhub.dev/google/elmo/2"
local_url = "module_elmo"   #when hub saved as module_elmo
elmo = hub.Module(url, trainable=False)

#get ELMO embeddings for each word in each sentence, returns ndarray of shape (#sentences, sentence_length, dim_length)
def elmoEmbedding(x):
    elmoEmbed = elmo(
    x,
    signature="default",
    as_dict=True)["elmo"]
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(elmoEmbed)
        session.close()
    return message_embeddings #returns ndarray, shape(#sentences,#max_words,#emb_dim)

def getWeightsPerSentence(i):
    outfname = 'data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/elmo' + str(FLAGS.year) + 'weight_' + str(i / 3) + '.txt'
    with open(outfname, 'w') as outputfile:
        print("sentence: " + str(i / 3) + " out of " + str(len(lines) / 3) + " in " + "raw_data;")
        # targets
        target = lines[i + 1].lower().split()
        # left and right context
        words = lines[i].lower().split()
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                words_l.append(word)
            else:
                words_r.append(word)
        # whole sentence
        sentence = [" ".join(words_l + target + words_r)]  # ELMO requires words, also e.g. a ';', to be separated by space
        # retrieve ELMO embeddings for each word in the sentence
        embedding_sent = elmoEmbedding(sentence)
        word_embeddings = embedding_sent[0, :, :]
        # save per word the embeddings to text file
        np.savetxt(outputfile, word_embeddings)
        return outfname

#combine train and test data
filenames=['data/programGeneratedData/300traindata' + str(FLAGS.year) + '.txt',
           'data/programGeneratedData/300testdata' + str(FLAGS.year) + '.txt']
with open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/raw_data' + str(FLAGS.year) + '.txt','w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
lines = open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/raw_data' + str(FLAGS.year) + '.txt').readlines()

#Save ELMo embedding for every sentence temporary to a separated text file, takes long, better in batches
temp_filenames = []
for i in range (0,len(lines),3):
    call_func = getWeightsPerSentence(i)
    temp_filenames.append(call_func)

#combine all temporary weight files for each sentence
with open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/ELMo' + str(FLAGS.year) + 'weights.txt','w') as outf:
    for tfname in temp_filenames:
        print(tfname)
        with open(tfname) as infile:
            for line in infile:
                outf.write(line)

#make table with unique words
lines = open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/raw_data' + str(FLAGS.year) + '.txt').readlines()
unique_words = []
unique_words_index = []
for i in range(0, len(lines), 3):
     target = lines[i + 1].lower().split()
     words = lines[i].lower().split()
     flag = True
     words_l, words_r = [], []
     for word in words:
         if word == '$t$':
             flag = False
             continue
         if flag:
             words_l.append(word)
         else:
             words_r.append(word)
     sentence = " ".join(words_l + target + words_r)
     words = sentence.split()
     for word in words:
         if word not in unique_words:
            unique_words.append(word)
            unique_words_index.append(0)

#make every word unique in the data and write to elmoAllVocab.txt
with open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/elmoAllVocab'+ str(FLAGS.year) +'.txt','w') as outFile:
    for i in range(0, len(lines), 3):
         print("sentence: " + str(i / 3) + " out of " + str(len(lines) / 3) + " in " + "raw_data;")
         target2 = lines[i + 1].lower().split()
         words2 = lines[i].lower().split()
         words_l2, words_r2 = [], []
         flag = True
         for word2 in words2:
             if word2 == '$t$':
                 flag = False
                 continue
             if flag:
                 words_l2.append(word2)
             else:
                 words_r2.append(word2)
         sentence2 = " ".join(words_l2 + target2 + words_r2)
         words3 = sentence2.split()
         for word3 in words3:
             index = unique_words.index(word3)      # get index in unique words table
             word_count = unique_words_index[index]
             unique_words_index[index] +=1
             item = str(word3) + '_' + str(word_count)
             outFile.write("%s\n" % item)

#change the raw text files to unique text files, corresponding with the embedding matrix
lines = open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/raw_data' + str(FLAGS.year) + '.txt').readlines()
vocab_old = open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/elmoAllVocab'+ str(FLAGS.year) +'.txt').readlines()
vocab = [x[:-1] for x in vocab_old]
counter = 0;
with open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/unique'+ str(FLAGS.year) +'allData.txt','w') as outFile:
    for i in range(0, len(lines), 3):
         print("sentence: " + str(i / 3) + " out of " + str(len(lines) / 3) + " in " + "raw_data;")
         words = lines[i].lower().split()
         target = lines[i + 1].lower().split()
         target_len = len(target)
         sentiment = lines[i + 2]

         sentence_data_unique = ""
         word_target_unique = ""
         dollar_count=0
         for word in words:
             if word[:4] == '$t$-' or word[:4] == '$t$.' \
                     or word[-4:] == '/$t$' or word[:4] == '$t$/' or word[:4] == '\'$t$':
                 sentence_data_unique += "$T$ " + vocab[counter] + " "
                 for j in range(0, target_len):
                     word_target_unique += vocab[counter] + " "
                     counter += 1
             if word == '$t$':
                 dollar_count += 1
                 if dollar_count == 1:
                     for j in range(0,target_len):
                         word_target_unique += vocab[counter] + " "
                         counter += 1
                 sentence_data_unique += "$T$ "
                 continue
             else:
                 sentence_data_unique += vocab[counter] + " "
                 counter += 1

         # write sentence_line in new file
         outFile.write(sentence_data_unique + '\n')

         #write target_line in new file
         outFile.write(word_target_unique + '\n')

         #write sentiment line in new file
         outFile.write(sentiment)

#Split in train and test file
linesAllData = open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/unique'+ str(FLAGS.year) +'allData.txt').readlines()
with open('data/programGeneratedData/' + str(FLAGS.embedding_dim) + 'traindata'+ str(FLAGS.year) +'.txt','w') as outTrain,\
        open('data/programGeneratedData/' + str(FLAGS.embedding_dim) + 'testdata'+ str(FLAGS.year) +'.txt','w') as outTest:
    for j in range(0, 5640):
        outTrain.write(linesAllData[j])
    for k in range(5640,len(linesAllData)):
        outTest.write(linesAllData[k])

#Combine Vocab and weights file in final Word-embedding matrix
with open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/elmoAllVocab'+ str(FLAGS.year) +'.txt') as ev:
  with open('data/temporaryData/temp_ELMo'+ str(FLAGS.year) +'/ELMo' + str(FLAGS.year) + 'weights.txt') as ew:
    with open("data/programGeneratedData/"+str(FLAGS.embedding_dim)+'ELMo_wordEmbedding'+ str(FLAGS.year) +'.txt',"w") as ep:
      evlines = ev.readlines()
      ewlines = ew.readlines()
      for i in range(len(evlines)):
          print(str(i) + " out of 45167")
          line = evlines[i].strip() + ' ' + ewlines[i]
          ep.write(line)
import re

f = open("LanguaL2017NumericalIndex.txt",encoding='ISO-8859-1')
food_words = set()
for line in f:
    food_words.update(line.split('\t')[-1].split('[')[0].split('(')[0].strip().split(' '))

    
f.close()
food_words = {str.lower(x) for x in food_words if len(x) > 2}
filter_words = [r'\d', 'ALPHA', 'BETA', 'GAMMA','BACTER']
for word in filter_words:
    food_words = {x for x in food_words if not re.search(word,x)}


food_words.add('ox')
#############################
f = open("FOOD_DES.txt", encoding='ISO-8859-1')
food_words_sr28 = set()
for line in f:
    food_words_sr28.update(line.split('~^~')[2].replace(',','').replace('(','').replace(')','').strip().split(' '))    


f.close()
food_words_sr28 = {str.lower(x) for x in food_words_sr28 if len(x) > 2}
filter_words = [r'\d']
for word in filter_words:
    food_words_sr28 = {x for x in food_words_sr28 if not re.search(word,x)}


f = open("food_words",'w')
food_words = {x for x in food_words.union(food_words_sr28) if len(x) > 2}
for phrase in food_words:
    if phrase[-1] == ',':
        phrase = phrase[:-1]
    words = phrase.split('/')
    for word in words:
        f.write(word)
        f.write('\n')

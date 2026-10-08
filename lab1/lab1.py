import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy3

nltk.download('punkt_tab', quiet=True)

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = sent_tokenize(text)
all_tokens = []
for sent in sentences:
    tokens = word_tokenize(sent)
    words = [t for t in tokens if t.isalpha()]
    all_tokens.append(words)

morph = pymorphy3.MorphAnalyzer()

def get_valid_analyses(word):
    """Возвращает список подходящих морфологических разборов слова"""
    results = []
    for parsed in morph.parse(word):
        pos = parsed.tag.POS
        
        if pos not in ('NOUN', 'ADJF'):
            continue
        
        if "Apro" in parsed.tag or "Anum" in parsed.tag:
            continue
        
        if parsed.score < 0.01:
            continue
        
        results.append({
            'lemma': parsed.normal_form,
            'pos': pos,
            'gender': parsed.tag.gender,
            'number': parsed.tag.number,
            'case': parsed.tag.case,
            'score': parsed.score,
        })
    return results

def check_match(a1, a2):
    """Функция проверки согласованности"""
    if a1['number'] != a2['number']:
        return False
    if a1['case'] != a2['case']:
        return False
    if a1['gender'] is not None and a2['gender'] is not None:
        if a1['gender'] != a2['gender']:
            return False
    return True

results = []

for words in all_tokens:
    for i in range(len(words) - 1):
        analyses1 = get_valid_analyses(words[i])
        analyses2 = get_valid_analyses(words[i + 1])
        
        if not analyses1 or not analyses2:
            continue
        
        best_pair = None
        best_score = -1
        
        for a1 in analyses1:
            for a2 in analyses2:
                if check_match(a1, a2):
                    combined_score = a1['score'] * a2['score']
                    if combined_score > best_score:
                        best_score = combined_score
                        best_pair = (a1['lemma'], a2['lemma'])

        if best_pair is not None:
            results.append(best_pair)

for lemma1, lemma2 in results:
    print(f"{lemma1} {lemma2}")

import gensim
import re

model = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)

target1 = "скафандр_NOUN"
target2 = "вентилятор_NOUN"

positive_words = ["гермошлем_NOUN", "вентиляция_NOUN"]
negative_words = []

results = model.most_similar(positive=positive_words, negative=negative_words, topn=20)

pat = re.compile(r"(.*)_NOUN")
filtered_results = []

for word, score in results:
    m = pat.match(word)
    if m:
        filtered_results.append((m.group(1), word, score))

top_10_nouns = filtered_results[:10]

print("Топ-10 наиболее близких существительных:")
found_targets = []

for i, (clean_word, full_word, score) in enumerate(top_10_nouns, 1):
    is_target = full_word in [target1, target2]
    print(f"{i:2d}. {clean_word}: {score:.4f}")
    
    if is_target:
        found_targets.append(full_word)

if target1 in found_targets and target2 in found_targets:
    print("\nЗаданная пара слов попала в топ-10.")
else:
    missing = [w for w in [target1, target2] if w not in found_targets]
    print(f"Не все целевые слова в топ-10. Не хватает слов: {missing}")

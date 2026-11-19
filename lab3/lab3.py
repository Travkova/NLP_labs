from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

# Инициализация модели
name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)

# Целевые слова
target1 = "машина"
target2 = "игра"

# Предложение для подбора окружения
sentence = "Эта [MASK] требует много времени и ресурсов."

# Функция получения топ-10 предсказаний
def get_top_10_predictions(sentence):
    input_data = tokenizer._encode_plus(sentence, return_tensors="pt")
    
    mask_index = torch.where(input_data["input_ids"][0] == tokenizer.mask_token_id)
    
    output = model(**input_data)
    
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)
    mask_word = softmax[0, mask_index[0], :]
    
    top = torch.topk(mask_word, 10)
    
    predictions = []
    for prob, token in zip(top[0][0].data, top[-1][0].data):
        word = tokenizer.decode([token]).strip()
        predictions.append((word, prob.item()))
    
    return predictions

# Вывод результата
print(f"Предложение: '{sentence}'")
print("\nТоп-10 наиболее вероятных слов на месте [MASK]:")

predictions = get_top_10_predictions(sentence)

for i, (word, prob) in enumerate(predictions, 1):
    is_target = (word == target1 or word == target2)
    marker = " <- ЦЕЛЕВОЕ СЛОВО" if is_target else ""
    print(f"{i:2d}. {word:20s} — {prob*100:.4f}%{marker}")

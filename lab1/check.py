import pymorphy3

morph = pymorphy3.MorphAnalyzer()

word = "характеристик"
parses = morph.parse(word)

print(f"Слово: '{word}'")
print(f"Всего вариантов разбора: {len(parses)}")
print("-" * 60)

for i, p in enumerate(parses):
    print(f"Вариант {i + 1}:")
    print(f"  Лемма:      {p.normal_form}")
    print(f"  Часть речи: {p.tag.POS}")
    print(f"  Полный тег: {p.tag}")
    print(f"  Род:        {p.tag.gender}")
    print(f"  Число:      {p.tag.number}")
    print(f"  Падеж:      {p.tag.case}")
    print(f"  Уверенность: {p.score:.3f}")
    print()

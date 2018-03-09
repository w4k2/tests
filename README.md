# Testy

Testujemy szesnaście zbiorów danych z UCI ML (w katalogu `datasets`). Stosujemy cztery podziały zbiorów:

- 10-fold CV
- 20-fold CV
- 30-fold CV
- 5 times 2-fold CV

Eksperymenty przeprowadzamy dla siedmiu klasyfikatorów:

- kNN (k=3),
- RBF SVM,
- Linear SVM,
- Naive Bayes,
- Decision Tree,
- Random Forest,
- MLP

Dodatkowo, dla każdego zbioru i każdej metody podziału dokonujemy dziesięć powtórzeń, w każdym przypadku ucząc i testując klasyfikatory na wspólnym podziale. W sumie daje to nam 78400 rezultatów klasyfikacji. Wyniki są składowane w katalogu `results`.

Ze względu na obecność tak problemów binarnych, jak wieloklasowych, w wyniku podawana jest miara *accuracy*, uzupełniona o odchylenie standardowe.

Dla każdej kombinacji metody podziału, powtórzenia i zbioru danych wyliczane są testy dla dwóch metod:

- test Wilcoxona,
- test T-Studenta.

Dla każdej metody przyjmujemy trzy progi:

- p = .9
- p = .95
- p = .99

W module analitycznym poszukiwane są kombinacje metody podziału, powtórzenia, użytego testu i progu, które dla każdego z klasyfikatorów dały co najmniej trzy zbiory danych, w których wybrany algorytm jest najlepszym lub należy do grupy maksymalnie trzech najlepszych zależnych od sobie statystycznie algorytmów (pod warunkiem, że każdy z nich jest statystycznie niezależny od tej samej liczby pozostałych).

W katalogu `plots` znajdują się wykresy słupkowe dla wszystkich wykrytych w ten sposób przypadków.

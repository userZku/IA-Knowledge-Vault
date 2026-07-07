# Fiche-pattern — Classer du texte (NLP)

> À garder sous le coude dès M2-B2, réutile en M8.
> Pendant « texte » de la [`fiche-pattern-ml-supervise.md`](../03_ML_Modelisation_Evaluation/fiche-pattern-ml-supervise.md).
> Le corps illustre la **classification de texte** ; pour la **reconnaissance
> d'entités (NER)**, voir l'encart « le delta ».
>
> **Le réflexe sobre** : pour classer du texte **quand on a des labels**, un
> `TF-IDF + régression logistique` est souvent **aussi performant — voire
> meilleur — sur un corpus métier de taille modérée** (quelques milliers
> d'exemples), et **toujours plus léger et plus explicable** qu'un gros modèle.
> On ne monte en pré-entraîné/LLM que si c'est justifié (cf. encart + panorama
> GenAI/LLM/RAG — *fiche à venir*).
>
> 🎯 **Réflexe baseline** : avant tout modèle plus complexe, entraîne **toujours**
> ce baseline simple et **mesure-le**. Il devient la **référence** que toute
> solution plus lourde (SLM, LLM) devra battre — en F1 **et** en coût/temps.

---

## La séquence en 6 étapes

```
   [1] Charger              [2] Explorer (mini)        [3] Découper
   textes + labels          longueur, équilibre        train_test_split
   (DataFrame)              des classes, exemples       stratify=y, rs=42
        │                         │                          │
        └─────────────────────────┴──────────────────────────┘
                                  ▼
   [4] Vectoriser + Entraîner    [5] Évaluer            [6] Persister
   Pipeline(TfidfVectorizer,     f1_score(macro)        joblib.dump(
            LogisticRegression)  classification_report    pipeline, ...)
```

| # | Étape | Geste | Fonction clé | Piège typique |
|---|---|---|---|---|
| 1 | **Charger** | textes + labels alignés | `pd.read_csv()` | Lignes texte/label décalées |
| 2 | **Explorer** | longueur des textes, équilibre des classes, **lire 2-3 exemples/classe** | `.value_counts()`, `.str.len()`, `df.groupby('label').head(3)` | Classes déséquilibrées **ou labels bruités/incohérents** non vus |
| 3 | **Découper** | train/test stratifié reproductible | `train_test_split(stratify=y, random_state=42)` | Oublier `stratify=y` |
| 4 | **Vectoriser + entraîner** | TF-IDF **dans le pipeline**, puis `fit` sur train | `TfidfVectorizer` + `LogisticRegression` | Vectoriser sur **tout** le corpus avant le split (fuite) |
| 5 | **Évaluer** | F1 macro + rapport par classe | `f1_score(average='macro')`, `classification_report` | Lire l'accuracy sur du déséquilibré |
| 6 | **Persister** | sauver le pipeline (**vectorizer + modèle**) | `joblib.dump(pipeline, ..., compress=3)` | Sauver le `clf` sans le `TfidfVectorizer` |

---

### Les 3 invariants à mémoriser

1. **Le vectorizer vit DANS le pipeline.** Le TF-IDF est ajusté sur le **train
   seulement** (`fit`) — l'inclure dans le `Pipeline` garantit qu'on ne « voit »
   pas le vocabulaire du test (sinon : fuite).
2. **On sauve le pipeline entier** (TF-IDF + modèle) — sinon impossible de
   re-transformer un nouveau texte en prod.
3. **F1 macro + rapport par classe**, jamais l'accuracy seule (les classes de
   texte sont presque toujours déséquilibrées).

---

## Le squelette minimal qui marche

```python
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

# [1] Charger — ici un mini-corpus ; remplace par ton CSV (colonnes texte, label)
textes = ["mon salaire de juin est faux", "prélèvement mutuelle erroné",
          "je pose mes congés en août", "inscription formation python", ...]
labels = ["paie", "mutuelle", "congés", "formation", ...]

# [3] Découper (stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    textes, labels, test_size=0.3, stratify=labels, random_state=42)

# [4] Vectoriser + entraîner (TF-IDF DANS le pipeline → pas de fuite)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42,
                               class_weight="balanced")),  # 'balanced' : à tester SI déséquilibre, pas par défaut
])
pipeline.fit(X_train, y_train)

# [5] Évaluer — F1 macro + rapport par classe + matrice de confusion
y_pred = pipeline.predict(X_test)
print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.3f}")
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)  # quelle classe confondue avec quelle autre ?

# [6] Persister (vectorizer + modèle ensemble)
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/text_clf_v1.joblib", compress=3)
```

---

### Comment l'adapter

| Tu veux changer… | Tu modifies… |
|---|---|
| Le corpus | L'étape [1] — charge ton CSV, garde `textes` et `labels` alignés |
| La représentation | L'étape [4] — `TfidfVectorizer(ngram_range=…, max_features=…, stop_words=…)` |
| Le modèle | L'étape [4] — `LinearSVC`, `MultinomialNB` ou `SGDClassifier` (les classiques sur TF-IDF ; **évite `RandomForest`** sur du creux haute dimension) |
| Le déséquilibre | `class_weight="balanced"` (LogReg/SVC) ou rééchantillonnage |
| Les métriques | L'étape [5] — `precision_score`/`recall_score` par classe selon le coût |

> 💡 **Prétraitement minimal d'abord** : `TfidfVectorizer` gère déjà la
> tokenisation, la casse (`lowercase=True`) et les accents
> (`strip_accents="unicode"`). N'ajoute **stemming / lemmatisation (spaCy, NLTK)
> que si un besoin le justifie** — pas par réflexe de notebook de 2018.

---

### Et si je n'ai pas de labels, ou besoin d'extraire ? (le delta)

**Trois questions suffisent à choisir** : *J'ai des labels ?* → classification
supervisée (ci-dessus). *Je veux extraire des entités ?* → **NER**. *Aucun
label ?* → **zero-shot**.

| Besoin | Approche | Outil |
|---|---|---|
| Classer **sans labels** | **zero-shot** (on définit les classes) | `pipeline("zero-shot-classification")` — cf. galerie modèles pré-entraînés notebook 02 |
| **Nuance forte / multilingue** + on a des labels | SLM **fine-tuné** | HuggingFace `transformers` |
| **Extraire des entités** (noms, lieux, n° sécu) | **NER** | `pipeline("token-classification")` — cf. galerie notebook 04, et M2-B2 (pseudonymisation) |

> 💡 **Garde-fou** : si tu as un **dataset labellisé** pour une classification, le
> `TF-IDF + LogReg` est le **point de départ sobre** — souvent imbattable en
> rapport qualité/coût. Le pré-entraîné se justifie surtout **sans labels** (zero-shot)
> ou pour une **vraie nuance** linguistique. cf. l'arbitrage M8-B2.

---

### Quand ça ne marche pas — diagnostic rapide

| Symptôme | Cause probable | Vérifier |
|---|---|---|
| F1 élevé au train, effondré au test | vocabulaire ajusté hors pipeline / sur-apprentissage | Le `TfidfVectorizer` est-il **dans** le `Pipeline` ? |
| Une classe à F1 = 0 | classe trop rare ou absente du train | `value_counts()` + `stratify` + `class_weight="balanced"` |
| Accents/casse créent des doublons de tokens | normalisation absente | `TfidfVectorizer(lowercase=True, strip_accents="unicode")` |
| Le `.joblib` rechargé plante sur un texte brut | tu as sauvé le `clf` sans le vectorizer | Sauver le **pipeline** complet |
| Beaucoup de mots ignorés à l'inférence | vocabulaire figé au `fit` (normal) | Attendu : un mot jamais vu au train est ignoré (**OOV** — une limite structurelle de TF-IDF ; c'est là que les *embeddings* / Transformers prennent le relais) |

---

*Fiche-pattern transverse — classification de texte. Pour la vision :
`fiche_pattern_vision.md` (*fiche à venir*). Pour le tabulaire,
[`fiche-pattern-ml-supervise.md`](../03_ML_Modelisation_Evaluation/fiche-pattern-ml-supervise.md).*

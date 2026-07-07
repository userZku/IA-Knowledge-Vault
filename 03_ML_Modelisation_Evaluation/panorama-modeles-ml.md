# Panorama des modèles de Machine Learning

> **Ressource transverse — à mobiliser dès M1, réutilisée en M2, M3, M4, M6, M7, M8.**
>
> Ce document pose **la carte du territoire** : qu'est-ce qu'un modèle de ML,
> quelles sont les grandes familles, comment elles se positionnent les unes
> par rapport aux autres. **Ce n'est pas un guide de choix** — le « comment
> choisir » se construit en **M4** avec la
> [`grille-decision-c4-reference.md`](../10_Architecture_Cadrage_Produit/grille-decision-c4-reference.md), enrichie collectivement.
>
> Objectif : que tu aies en tête, dès M1, où se situe un `RandomForest` dans
> le paysage ML, et que tu ne confondes plus *régression linéaire*,
> *réseau de neurones* et *LLM*.
>
> Lecture : ~25 min. Pas d'exercice — un document de référence.

---

## 1. Qu'est-ce qu'un modèle de ML ?

Un **modèle de Machine Learning** est une **fonction paramétrée** qui
transforme une entrée `X` en une prédiction `ŷ` :

```
ŷ = f(X ; θ)
```

- `X` : les **features** (variables d'entrée — colonnes du dataset).
- `θ` : les **paramètres internes** du modèle (poids d'une régression
  linéaire, seuils des arbres de décision, poids d'un réseau de neurones…).
  Ce sont eux qui sont **appris** pendant l'entraînement.
- `ŷ` : la **prédiction** (classe, score, valeur continue, séquence de
  tokens…).

**Ce qui distingue un modèle de ML d'un programme classique** : tu ne
codes pas explicitement la règle métier. Tu fournis des **exemples
annotés** (couples `X`, `y`), et l'algorithme d'apprentissage **ajuste les
paramètres `θ`** pour minimiser une fonction d'erreur entre `ŷ` et `y`.

> 💡 **Paramètres vs hyperparamètres** — distinction structurante :
> - Les **paramètres** (θ) sont **appris** automatiquement par `model.fit()`.
> - Les **hyperparamètres** sont **choisis avant** l'entraînement
>   (nombre d'arbres, profondeur max, taux d'apprentissage…). Ils
>   conditionnent **comment** l'apprentissage se déroule.

### Le cycle de vie minimal d'un modèle

```
        DONNÉES                      ENTRAÎNEMENT                INFÉRENCE
   ┌──────────────┐              ┌─────────────────┐         ┌───────────┐
   │ X_train      │  ───────►    │ model.fit(X, y) │  ──►    │ model.    │
   │ y_train      │              │   ajuste θ      │         │ predict(X)│
   └──────────────┘              └─────────────────┘         └───────────┘
                                          │                        │
                                          ▼                        ▼
                                  Modèle persisté            Prédictions ŷ
                                  (.joblib + méta)
```

Ce cycle est **commun à toutes les familles** de modèles — c'est même la
force de scikit-learn : `fit` / `predict` / `score` sont les mêmes
appels, que tu utilises une régression logistique ou un Random Forest.

---

## 2. Les trois régimes d'apprentissage

| Régime | Idée | Vu dans le parcours |
|---|---|---|
| **Supervisé** | On apprend à partir d'exemples **étiquetés** (X, y). Le modèle apprend la relation entrée → cible. | **M1, M2, M4, M6, M7, M8** |
| **Non-supervisé** | Pas d'étiquette. On cherche **structure interne** : groupes (clustering), réduction de dimension, détection d'anomalie. | **M3** |
| **Renforcement** | Un agent apprend par **essai/erreur** dans un environnement (récompense différée). | Hors parcours |

> 💡 **Cas hybride : auto-supervisé** — utilisé massivement par les LLM et
> les *foundation models*. On invente une tâche prétexte à partir des données
> brutes (prédire le mot suivant, masquer un patch d'image…). On le revoit
> en **M7-M8**.

Dans tout le parcours ATOS Parcours 2, **on reste majoritairement en
supervisé**. C'est le geste pro le plus courant en intégration IT.

---

## 3. Supervisé : classification vs régression

Deux sous-régimes selon la **nature de la cible `y`** :

| Sous-régime | `y` est… | Exemples métier | Métriques typiques |
|---|---|---|---|
| **Classification** | catégoriel (binaire ou multi-classe) | Risque crédit (défaut / pas défaut), tri ticket (5 catégories), détection fraude | F1, ROC-AUC, matrice de confusion, accuracy, precision/recall |
| **Régression** | continu | Prix immobilier, consommation énergétique, latence prévue | MAE, RMSE, R², MAPE |

> **M1-B1 traite un cas de classification binaire** (risque de défaut
> Pyrenex Crédit). M4-B1 verra une régression (consommation énergétique).

Une **3ᵉ variante moins fréquente** : la **classification multi-label**
(une observation peut avoir plusieurs étiquettes — tags d'un article,
symptômes d'un patient). On n'y touche pas en M1 mais sache que ça existe.

---

## 4. Les grandes familles de modèles supervisés tabulaires

C'est **la** carte mentale à avoir en tête en M1. Cinq familles, leurs
forces et leurs limites. **Aucun "meilleur" dans l'absolu** — chacune
brille selon le contexte (cf. `grille-decision-c4-reference.md`).

### 4.1 Modèles linéaires

**Exemples** : `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`.

**Idée** : la prédiction est une **combinaison linéaire des features**
(ŷ = β₀ + β₁·x₁ + β₂·x₂ + … ).

| Force | Limite |
|---|---|
| Très **explicable** (coefficients lisibles) | Hypothèse de linéarité — limite si interactions complexes |
| Entraînement **rapide**, peu de données suffisent | Sensible aux features non normalisées, aux outliers |
| Référence réglementaire (crédit, santé) | Performance plafonnée sur signaux non-linéaires |

**Quand y penser** : peu de données (< 1 k), besoin d'explicabilité fort,
**baseline** systématique à essayer avant de complexifier.

### 4.2 Voisinage (kNN)

**Exemples** : `KNeighborsClassifier`, `KNeighborsRegressor`.

**Idée** : pour prédire un point, on regarde **les k voisins les plus
proches** dans le dataset d'entraînement et on agrège leur cible.

| Force | Limite |
|---|---|
| Très simple à comprendre, pas d'entraînement | Inférence lente sur gros datasets |
| Pas d'hypothèse sur la forme du signal | Sensible à la « malédiction de la dimension » |
| Performant en faible dimension | Nécessite normalisation soignée |

**Quand y penser** : peu de features, signal local fort, prototype rapide.

### 4.3 Modèles probabilistes (Naive Bayes)

**Exemples** : `GaussianNB`, `MultinomialNB`, `BernoulliNB`.

**Idée** : on applique le théorème de Bayes en supposant les features
**indépendantes** conditionnellement à la classe (l'hypothèse « naïve »).

| Force | Limite |
|---|---|
| Très rapide, scale bien sur gros volumes texte | L'hypothèse d'indépendance est rarement vraie |
| Robuste avec peu de données | Performance limitée hors texte / spam |
| Naturellement probabiliste | Pas d'interactions modélisées |

**Quand y penser** : classification de texte, spam, baseline NLP rapide.

### 4.4 Arbres et ensembles ← **C'est ici que vit `RandomForest`**

**Exemples** :
- **Arbre simple** : `DecisionTreeClassifier`, `DecisionTreeRegressor`.
- **Ensembles bagging** : `RandomForestClassifier`, `RandomForestRegressor`.
- **Ensembles boosting** : `GradientBoostingClassifier`, `HistGradientBoostingClassifier`, XGBoost, LightGBM, CatBoost.

**Idée arbre simple** : on apprend une **série de questions binaires
sur les features** (« revenu < 30 k ? oui → branche gauche »). Les
feuilles de l'arbre portent la prédiction.

**Idée ensemble** : un seul arbre est instable et sur-apprend. On en
combine plusieurs centaines :
- **Bagging** (Random Forest) : on entraîne en parallèle N arbres sur
  des sous-échantillons aléatoires, on vote / moyenne. **Réduit la variance**.
- **Boosting** (Gradient Boosting) : on entraîne séquentiellement, chaque
  nouvel arbre corrige les erreurs du précédent. **Réduit le biais**, plus
  performant en général mais plus délicat à tuner.

| Force | Limite |
|---|---|
| **Top sur tabulaire hétérogène** (mix num + catégoriel) | Moins lisible qu'un linéaire (mais SHAP donne l'explicabilité) |
| Robuste aux outliers, peu sensible à la normalisation | Modèles lourds (.joblib parfois > 100 Mo) |
| Gère les non-linéarités et interactions naturellement | Tuning des hyperparamètres peut être coûteux |

**Quand y penser** : **données tabulaires** entre 1 k et 100 k lignes,
performance demandée mais explicabilité possible via SHAP. **C'est le
choix par défaut sur Lending Club en M1-B1.**

> 📌 **Pourquoi Random Forest a été choisi par Pyrenex en 2017 ?**
> Lending Club est un dataset **tabulaire mixte** (numérique + catégoriel),
> de taille moyenne (30 k), avec des **interactions non-linéaires**
> (le FICO seul ne suffit pas — il interagit avec l'emploi, le montant…),
> et le métier crédit accepte un **modèle d'ensemble + explicabilité SHAP**.
> Tu retrouveras ce raisonnement dans la `grille-decision-c4-reference.md`.

### 4.5 Réseaux de neurones (MLP / Deep Learning)

**Exemples scikit-learn** : `MLPClassifier`, `MLPRegressor` (petits réseaux).
**Frameworks dédiés** : PyTorch, TensorFlow/Keras (réseaux profonds, vision,
NLP).

**Idée** : des **couches de neurones** non-linéaires, chacune apprenant une
représentation transformée des features. Avec assez de couches et de données,
**approxime n'importe quelle fonction**.

| Force | Limite |
|---|---|
| Puissance d'expression maximale | Nécessite **beaucoup de données** (généralement > 100 k) |
| Indispensable hors-tabulaire (image, texte, son) | Boîte noire (XAI complexe) |
| Bibliothèques matures (PyTorch, TF) | Coût d'entraînement (GPU) et de maintenance |

**Quand y penser** : signal structurel (image, texte, audio), gros volumes,
ou **transfer learning** depuis un foundation model.

> ⚠️ **Anti-pattern fréquent** : utiliser un réseau de neurones sur du
> tabulaire 30 k lignes quand un Gradient Boosting fait mieux en 5 min
> d'entraînement. Le réflexe « deep learning everywhere » est l'un des
> tropismes qu'on **combat** dans le parcours ATOS.

---

## 5. Au-delà du tabulaire : le signal structurel

Quand `X` n'est plus une table mais une **image, du texte, du son ou une
séquence temporelle**, les familles classiques ne suffisent plus. Survol
de ce qu'on rencontre dans le parcours :

### Vision (M4-B2)

- **CNN** (réseaux convolutifs) : ResNet, EfficientNet — adaptés à
  l'image, exploitent la localité spatiale.
- **Vision Transformers (ViT)** : transposent l'attention du NLP à l'image.
- **Transfer learning** : on prend un modèle pré-entraîné sur ImageNet,
  on ré-entraîne les dernières couches sur **ton** problème (geste pro
  central en industrie).

### Texte et séquence (M7, M8)

- **Embeddings** : représenter un mot ou un document comme un vecteur
  dense (Word2Vec, GloVe, sentence-transformers).
- **Transformers** : architecture qui a remplacé les RNN/LSTM depuis 2017,
  base de tous les LLM modernes.
- **LLM** (Large Language Models) : GPT, Claude, Llama, Mistral. Modèles
  pré-entraînés sur des téraoctets de texte, mobilisables en *zero-shot*,
  *few-shot* ou *fine-tuning*.
- **SLM** (Small Language Models) : versions distillées (DistilBERT,
  Phi-3, Mistral 7B). Compromis perf/coût/sobriété.

### Foundation models et transfer learning

Concept **transverse** depuis 2018-2020 : un modèle **pré-entraîné massivement**
sur des données génériques (texte, image, multimodal), puis **adapté** à un
problème spécifique via :
- **Zero-shot** : on l'utilise tel quel avec un *prompt* (LLM, CLIP).
- **Few-shot / in-context learning** : on lui donne quelques exemples
  dans le prompt.
- **Fine-tuning** : on ré-entraîne tout ou partie des poids sur **ton**
  dataset.
- **LoRA / adapters** : fine-tuning **léger**, on n'apprend qu'une petite
  fraction des poids (compromis performance / coût).

> 🎯 **Architecture émergente : le RAG** (Retrieval-Augmented Generation) —
> on couple un LLM (génération) à une base de connaissances vectorielle
> (récupération). Brief central en **M7-B2**.

---

## 6. Comment ce panorama est mobilisé dans le parcours

| Module | Familles principalement vues | Geste pédagogique |
|---|---|---|
| **M0** | Aucune entraînée — on **consomme** une API IA tierce (HuggingFace) | Acculturation, intégration |
| **M1** | **Arbres / ensembles** (Random Forest) | Réentraîner et packager |
| **M2** | Linéaire, Random Forest, Gradient Boosting | Préparation données, classes déséquilibrées, équité |
| **M3** | **Non-supervisé** (clustering, réduction de dim) | Préparation données, exploration |
| **M4** | **Toutes les familles** + foundation models en option | **Choisir** (grille C4) |
| **M5** | Le modèle entraîné en M1 (ou plus tard) | Déploiement, monitoring |
| **M6** | Boosting + réentraînement | Performance, drift, ré-entraînement |
| **M7** | LLM, RAG, embeddings | Architecture avancée |
| **M8** | Agents LLM, foundation models multimodaux | Architecture experte |

> 💡 **La capacité à choisir éclairé entre ces familles** est le cœur
> pédagogique du parcours. Tu n'as pas à la maîtriser en M1 — on construit
> la `grille-decision-c4-reference.md` progressivement.

---

## 7. Vocabulaire express

| Terme | Définition courte |
|---|---|
| **Algorithme** | Méthode mathématique d'apprentissage (Random Forest, régression logistique…). |
| **Modèle entraîné** | Résultat de l'application d'un algorithme à des données — contient les paramètres `θ` appris. |
| **Famille de modèles** | Groupe d'algorithmes partageant une logique commune (linéaires, arbres, réseaux…). |
| **Architecture** | Pour les réseaux de neurones : la structure des couches (CNN, Transformer, MLP…). |
| **Pre-trained model** | Modèle déjà entraîné sur un large corpus, fourni clé en main. |
| **Fine-tuning** | Ré-entraînement ciblé d'un modèle pré-entraîné sur **ton** problème. |
| **Transfer learning** | Stratégie générale de réutilisation des connaissances d'un modèle sur un nouveau problème. |
| **Baseline** | Modèle simple servant de point de comparaison — souvent un linéaire ou un arbre nu. |
| **SOTA** (*State of the Art*) | Meilleure performance publiée sur un benchmark — pas forcément le bon choix en industrie. |
| **No Free Lunch theorem** | Résultat théorique : aucun algorithme n'est universellement meilleur. Le contexte décide. |

---

## 8. Pour aller plus loin (optionnel)

- 📘 Aurélien Géron, *Machine Learning avec scikit-Learn* (3ᵉ éd.) — chapitres 1
  (panorama), 4 (linéaires), 6 (arbres), 7 (ensembles).
- 📘 *LLM Engineer's Handbook* — pour M7/M8 (architectures avancées).
- 🌐 [scikit-learn — *Choosing the right estimator*](https://scikit-learn.org/stable/machine_learning_map.html) — la carte officielle, un peu datée mais utile.
- 🌐 [Papers With Code — benchmarks par tâche](https://paperswithcode.com/) — pour voir qui est SOTA sur quoi.
- 🎥 [Andrej Karpathy — *Neural Networks: Zero to Hero*](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (YouTube) — pour comprendre les réseaux de neurones de zéro.

---

*Panorama des modèles ML — ressource publique transverse, parcours IA ATOS.*

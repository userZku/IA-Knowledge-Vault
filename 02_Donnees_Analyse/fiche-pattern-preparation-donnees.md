# Fiche-pattern — Auditer, préparer et industrialiser un jeu de données

> À garder sous le coude de M2 jusqu'à M3 (et au-delà :
> toute mission commence par préparer la donnée).
> Le geste est le même quel que soit le dataset — seuls les détails changent.

---

## La séquence en 6 étapes

```
   [1] Découvrir            [2] Auditer la qualité     [3] Auditer l'éthique
   ─────────────            ────────────────────       ─────────────────────
   pd.read_csv              .isna().sum()              équilibre de la cible
   schéma + types           .dtypes / .describe()      disparate impact (DI)
   repérer variables        outliers, doublons         règle des 4/5
   sensibles                ≥4 viz ciblées             DI < 0.8 → signal
        │                         │                          │
        └─────────────────────────┴──────────────────────────┘
                                  ▼
   [4] Choisir le           [5] Industrialiser         [6] Adapter
       prétraitement        ────────────────           ─────────────
   ─────────────            ColumnTransformer          besoin d'évoluer ?
   imputation ?             + Pipeline                 ajoute / retire /
   encodage ?               joblib.dump(               modifie une branche,
   normalisation ?            pipe, path)              re-fit (pas tout)
   (justifier chaque choix)
```

> Les visualisations ne sont **pas décoratives** : chacune **répond à une
> question** (répartition, manquants, distribution de la cible, corrélations).
> « ≥4 » est un **plancher**, pas un objectif — 4 graphes « pour faire 4 graphes »
> ne valent rien.

| # | Étape | Geste | Fonction clé | Piège typique |
|---|---|---|---|---|
| 1 | **Découvrir** | Charger, lire le schéma, repérer les variables sensibles | `pd.read_csv()`, `.dtypes`, `.head()` | Ne pas repérer une variable sensible composite (ex. `personal_status_sex`) |
| 2 | **Auditer la qualité** | Manquants, types, outliers, cohérence | `.isna().sum()`, `.describe()`, `.value_counts()` | Dire « il y a des manquants » sans chiffrer ni décider quoi en faire |
| 3 | **Auditer l'éthique** | Déséquilibre cible + disparate impact | `groupby` + ratio des taux de succès | Calculer le DI sur le mauvais groupe / le mauvais outcome |
| 4 | **Choisir** | Imputation / encodage / scaling, **justifiés** | (décision, pas de code) | Encoder une variable **ordinale** en OneHot → perte de l'ordre |
| 5 | **Industrialiser** | Tout dans un `Pipeline` + `ColumnTransformer`, persisté | `ColumnTransformer`, `Pipeline`, `joblib.dump` | Transformations Pandas éparpillées, non rejouables en prod |
| 6 | **Adapter** | Faire **évoluer** le pipeline : ajouter/retirer une branche, changer l'imputation ou l'encodage | re-`fit` du pipeline | Refaire tout le pipeline de zéro au lieu d'ajuster la bonne branche |

---

### Les 4 invariants à mémoriser

1. **On documente les biais, on ne les corrige pas (à ce stade).** Détecter, chiffrer, alerter le client. La mitigation viendra plus tard (M7).
2. **On industrialise dans un Pipeline, pas en Pandas éparpillé.** Un `ColumnTransformer` + `Pipeline` est réutilisable, persistable, testable — des `df["x"] = ...` dispersés ne le sont pas.
3. **L'ordre d'une variable ordinale se préserve.** `OrdinalEncoder(categories=[...])` avec l'ordre explicite, pas `OneHotEncoder` (qui perd la hiérarchie basic < plus < premium).
4. **`handle_unknown="ignore"` sur OneHot.** Sinon une catégorie jamais vue à l'entraînement fait planter la prédiction en prod.
5. **La cible `y` ne passe JAMAIS dans le préprocesseur.** On sépare `X` (features) de `y` (cible) **avant** tout `ColumnTransformer` — sinon on « prépare » la réponse : fuite de données garantie.
6. **On adapte le prétraitement aux données, pas l'inverse.** Pas de `OneHot` / `StandardScaler` / `SimpleImputer` « partout par défaut » : chaque branche se justifie par la **nature** de la variable.

---

## Le squelette minimal qui marche

```python
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib

df = pd.read_csv("data/raw.csv")

# [3] Disparate impact (règle des 4/5) — à la main, en Pandas
def disparate_impact(df, sensible, target, positive):
    sr = df.groupby(sensible)[target].apply(lambda x: (x == positive).mean())
    if sr.max() == 0:                       # garde-fou : aucun cas positif → DI indéfini
        return None, sr
    di = sr.min() / sr.max()
    print(f"{sensible}: DI = {di:.3f}  ({'⚠️ SIGNAL' if di < 0.8 else '✅ OK'})")
    return di, sr

# [4]+[5] Industrialiser : 3 branches selon la nature des features
NUM = ["credit_amount", "age"]
ORD = {"savings_account": ["<100", "100-500", "500-1000", ">=1000", "unknown"]}
CAT = ["purpose", "housing"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OrdinalEncoder(categories=list(ORD.values())))]), list(ORD)),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])

TARGET = "credit_risk"
X = df[NUM + list(ORD) + CAT]   # features uniquement
y = df[TARGET]                  # la cible reste DEHORS du préprocesseur (invariant 5)
X_clean = preprocessor.fit_transform(X)
print("shape après préparation :", X_clean.shape)   # + de colonnes (one-hot)

# 🔭 En M4, ce préprocesseur deviendra la 1re étape d'un pipeline complet :
#   pipe = Pipeline([("prep", preprocessor), ("model", ...)]).fit(X, y)
# Ici (M2 = C3 préparer), on s'arrête à la préparation — le modèle vient plus tard.

# [5] Persister le pipeline (fin de l'industrialisation — pas le DataFrame transformé)
Path("src").mkdir(exist_ok=True)
joblib.dump(preprocessor, "src/pipeline.joblib", compress=3)
```

---

### Comment l'adapter

| Tu veux changer… | Tu modifies… |
|---|---|
| Le dataset | L'étape [1] — `pd.read_csv("mon_fichier.csv")` et les 3 listes NUM / ORD / CAT |
| La stratégie d'imputation | L'étape [4] — `SimpleImputer(strategy="mean"/"median"/"most_frequent")` ou `KNNImputer()` |
| L'encodage d'une variable | Déplace-la entre les listes ORD (ordonnée) et CAT (non ordonnée) |
| Ajouter / retirer une variable | L'étape [6] **adapter** — déplace-la dans/hors la bonne liste (NUM/ORD/CAT, définies au bloc `[4]+[5]`), re-`fit`, vérifie la nouvelle shape |

---

### Quand ça ne marche pas — diagnostic rapide

| Symptôme | Cause probable | Vérifier |
|---|---|---|
| `ValueError: could not convert string to float` | Une catégorielle est restée hors du `ColumnTransformer` | Toutes les colonnes objet sont-elles dans ORD ou CAT ? |
| Des NaN subsistent après transformation | Imputation manquante sur une branche | Chaque branche a-t-elle un `SimpleImputer` ? |
| Une catégorie de prod fait planter | `OneHotEncoder` sans `handle_unknown` | Ajouter `handle_unknown="ignore"` |
| Le DI semble absurde (> 1 ou très petit) | Mauvais groupe/outcome positif | Le « positif » est-il bien l'outcome favorable (ex. `good_credit`) ? |
| L'ordre d'une variable ordinale est perdu | Encodée en OneHot | Passer en `OrdinalEncoder(categories=[...])` avec l'ordre explicite |
| Le `.joblib` rechargé ne transforme pas pareil | On a sauvé le DataFrame, pas le pipeline | Sauver le `ColumnTransformer`/`Pipeline`, pas `X_clean` |

---

### Les 3 questions du ML Engineer

1. **Puis-je faire confiance à ces données ?** → l'**audit** (étapes 1-3)
2. **Comment les rendre exploitables ?** → la **préparation justifiée** (étape 4)
3. **Comment rejouer exactement le même traitement demain ?** → le **pipeline persisté** (étapes 5-6)

---

*Fiche-pattern transverse — préparation de données. La séquence audit → choix →
industrialisation → adaptation se réutilise à chaque nouvelle source (M2, M3…).*
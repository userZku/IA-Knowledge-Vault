# Grille de décision C4 — Choisir un modèle IA

> **Livrable pédagogique majeur du parcours ATOS.**
> **Amorce** posée par la formatrice (Axes 1-2). Les **Axes 3-4, la cartographie
> et les cas types se construisent COLLECTIVEMENT** en restitution **M4-B1**
> (mercredi midi), à partir de vos benchmarks et de vos *decision cards*.
> Ensuite enrichie à chaque module (M4-B2, M7-B1, M7-B2, M8-B1), puis
> réutilisée pour le **notebook certif M9**.
>
> **Statut** : 🔨 **amorce à compléter** — les zones marquées *« à construire
> en restitution »* sont vides **exprès**. C'est vous qui les remplissez.
>
> 📦 **Compagnon** : pour la décision **en amont** (où/comment stocker les
> données), voir [`grille-decision-stockage.md`](../07_Data_Engineering/grille-decision-stockage.md).

---

## 🎯 À quoi sert cette grille ?

Face à un besoin métier nouveau, **quel modèle choisir** ? La grille te
donne une **première lecture** structurée selon 4 axes :

1. **Volume de données disponible** (du peu au beaucoup)
2. **Complexité du signal** (linéaire ↔ non-linéaire ↔ structurel)
3. **Contraintes métier** (explicabilité, latence, coût, sobriété)
4. **Maintenance attendue** (réentraînement fréquent ou non, MLOps requis)

**Important** : la grille n'est **pas un oracle**. C'est un **support
de raisonnement** : tu te poses chaque axe, tu **justifies** ton choix
dans ton notebook (M4, M7, M8, certif). Une case ne dit jamais *« le
meilleur modèle »* mais *« le meilleur SI tel critère prime »*.

---

## 📊 La grille

### Axe 1 — Volume de données · *amorce à challenger*

| Volume | Familles candidates |
|---|---|
| **< 1 k échantillons** | Linéaire (LinearRegression / Ridge), Arbre simple, *Zero-shot foundation model* (CLIP / GPT-4) |
| **1 k - 100 k** | Random Forest, Gradient Boosting (HGB, XGBoost), Transfer learning |
| **> 100 k** | Boosting tuné, Réseau de neurones, Transfer learning, Foundation model fine-tuné |
| **> 1 M** | Deep learning (custom), Transformers, distillation possible |

### Axe 2 — Complexité du signal · *amorce à challenger*

| Complexité | Familles candidates |
|---|---|
| **Linéaire** (corrélation directe features ↔ cible) | Linéaire (Ridge, Lasso), arbre simple en challenge |
| **Non-linéaire faible** (interactions modérées) | Random Forest, Gradient Boosting |
| **Non-linéaire forte** (interactions complexes) | Boosting tuné, Réseau de neurones |
| **Structurel** (image, texte, séquence) | CNN, Transformer, Transfer, Foundation model |

### Axe 3 — Contraintes métier · 🔨 *à construire en restitution*

> On remplit cet axe **ensemble**, à partir des contraintes de vos *decision
> cards* et de vos **mesures réelles** de M4-B1 (temps d'inférence, temps de
> train, explicabilité constatée). Une ligne d'exemple est donnée par
> sous-tableau ; **complétez les autres**.

#### Explicabilité

| Besoin | Familles privilégiées |
|---|---|
| **Décision réglementaire** (crédit, santé, RH) | Linéaire, Arbre simple (lisible) |
| **Confort utilisateur** (peut être complexe) | *(à remplir)* |
| **Boîte noire acceptable** | *(à remplir)* |

#### Latence d'inférence

| Contrainte | Familles privilégiées |
|---|---|
| **< 10 ms** (temps réel critique) | *(à remplir — vos mesures ms/1k éch.)* |
| **10 ms - 100 ms** | *(à remplir)* |
| **100 ms - 1 s** | *(à remplir)* |
| **> 1 s acceptable** | *(à remplir)* |

#### Coût / sobriété

| Budget | Familles privilégiées |
|---|---|
| **Train quasi nul / Inférence quasi nulle** | *(à remplir)* |
| **Train modéré / Inférence faible** | *(à remplir)* |
| **Train lourd (GPU) / Inférence coûteuse** | *(à remplir)* |
| **API par requête** | *(à remplir)* |

> 💡 Si une dimension vous manque ici (empreinte carbone, RGPD, explicabilité
> réglementaire ACPR, time-to-market…), **ajoutez-la** : c'est le geste attendu.

### Axe 4 — Maintenance attendue · 🔨 *à construire en restitution*

| Fréquence réentraînement | Familles privilégiées |
|---|---|
| **Jamais** (modèle statique) | Linéaire, Foundation model zero-shot |
| **Trimestrielle** (drift modéré) | *(à remplir)* |
| **Mensuelle** (drift fort) | *(à remplir — lien MLOps M5/M6)* |
| **Continu / online** | *(à remplir)* |

---

## 🗺️ Cartographie des modèles par contexte · 🔨 *à qualifier ensemble*

> Tableau à **remplir en restitution** à partir de vos benchmarks M4-B1 et de la
> grille ci-dessus. Une ligne est donnée en exemple.

| Modèle | Volume idéal | Explicabilité | Latence | Train rapide |
|---|---|---|---|---|
| **LinearRegression / Ridge** | < 100 k | ✅✅✅ | < 1 ms | ✅✅✅ |
| **RandomForest** | *(à remplir)* | *(à remplir)* | *(à remplir)* | *(à remplir)* |
| **HistGradientBoosting** | *(à remplir)* | *(à remplir)* | *(à remplir)* | *(à remplir)* |
| **CNN from scratch** | *(à compléter en M4-B2)* | | | |
| **Transfer learning (ResNet, ViT)** | *(à compléter en M4-B2)* | | | |
| **Zero-shot CLIP** | *(à compléter en M4-B2)* | | | |
| **LLM via API** | *(à compléter en M7)* | | | |

---

## 💼 Cas types pédagogiques · 🔨 *construits au fil du parcours*

> Chaque cas = un besoin réel placé sur les 4 axes + le verdict chiffré. On
> construit le **premier ensemble** en restitution M4-B1.

### Cas M4-B1 (Bike Sharing — régression saisonnière, ~17 k lignes) · *à construire*

- **Volume** : *(à placer)*
- **Complexité** : *(à placer)*
- **Explicabilité** : *(à placer)*
- **Latence** : *(à placer)*
- **Maintenance** : *(à placer)*
- **Verdict type de la promo** : *(à trancher collectivement — avec le chiffre)*

### Cas M4-B2 (PCB Defect — vision industrielle) · *à construire après M4-B2*

### Cas M7 (santé) · *à compléter en M7*

### Cas M8 (cas tirés) · *à compléter en M8*

---

## 🧠 Méthode d'utilisation

Pour chaque nouveau besoin métier :

1. **Place le cas sur les 4 axes** (volume / complexité / contraintes /
   maintenance)
2. **Identifie 2-3 familles candidates** qui cochent les axes critiques
3. **Benchmark sur même split + mêmes métriques** (règle d'or
   comparabilité issue de M1)
4. **Choisis** en justifiant chiffré (vs *« j'aime ce modèle »*)
5. **Documente** la décision dans ton notebook avec un renvoi à cette grille

---

## 🔁 Évolution de la grille

| Version | Semaine | Modifications | Auteur·rice |
|---|---|---|---|
| amorce | M4 (avant restit.) | Axes 1-2 posés, reste à trous | Marianne (formatrice) |
| v1.0 | M4 (restit. B1) | Axes 3-4 + cartographie + cas Bike Sharing construits collectivement | Promo ATOS G1 |
| v1.1 | M4 (fin) | Ajout cas vision PCB après M4-B2 | Promo ATOS G1 |
| v2.0 | M7 | Ajout familles foundation models / LLM / RAG | Promo ATOS G1 |
| v2.1 | M8 | Ajout cas par tirage | Promo ATOS G1 |
| v3.0 | À voir | Consolidation finale | — |

> 💡 **La grille n'est jamais terminée.** Chaque cas nouveau enrichit
> notre lecture. La **v1.0 se construit collectivement** en mercredi M4-B1 —
> pas par la formatrice seule.

---

## 📚 Pour aller plus loin

- **scikit-learn — *Choosing the right estimator*** : <https://scikit-learn.org/stable/machine_learning_map.html>
  (carte officielle, à connaître mais plus restreinte que cette grille)
- **AWS — *ML Decision Tree*** : <https://aws.amazon.com/blogs/machine-learning/which-aws-machine-learning-service-to-use/>
  (centré cloud, pertinent en M5+)
- **HuggingFace — *Choosing the right model*** : <https://huggingface.co/docs/transformers/index>
  (centré transformers, pertinent en M4-B2 / M7)

---

*Grille pédagogique ATOS — co-construite avec la promo. Amorce Axes 1-2 par
Marianne Arrué (formatrice) ; Axes 3-4, cartographie et cas types construits en
restitution M4-B1 avec la promo G1.*

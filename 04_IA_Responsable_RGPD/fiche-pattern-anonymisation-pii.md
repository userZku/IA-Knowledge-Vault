# Fiche-pattern — Anonymiser du texte libre (PII)

> À garder sous le coude dès M2-B2, réutile en M3
> (RGPD multi-sources) et M7 (NLP/RAG). Complète l'encart NER de
> [`fiche-pattern-texte-nlp.md`](../05_NLP_Texte_PII/fiche-pattern-texte-nlp.md).
>
> **Le réflexe** : on neutralise d'abord ce qui est **déterministe** (email,
> téléphone, IBAN → **regex**), **puis** ce qui est **probabiliste** (noms,
> lieux → **NER spaCy**). Jamais l'inverse. Et **pseudonymisation ≠
> anonymisation** : si tu gardes une table de correspondance, c'est réversible
> (donnée perso, toujours RGPD). Sans table, c'est *un pas vers*
> l'anonymisation — **condition nécessaire mais pas suffisante** (un résiduel
> peut ré-identifier : « le PDG de [ORG] »).
>
> ⚠️ **Règle d'or** : on anonymise **en amont** du pipeline — avant feature
> engineering, entraînement ou envoi à un service tiers (API LLM incluse). Une
> fois la donnée personnelle partie en aval, il est trop tard.

---

## La séquence en 6 étapes

```
   [1] Charger             [2] Repérer les PII        [3] Choisir la stratégie
   texte (DataFrame)       regex : EMAIL/PHONE/IBAN   redaction / substitution
   colonne libre           NER : PERSON/GPE (ORG=métier)  hash / généralisation
        │                         │                          │
        └─────────────────────────┴──────────────────────────┘
                                  ▼
   [4] Anonymiser          [5] Vérifier               [6] Documenter
   regex AVANT NER         ~10 cas avant/après        limites (faux + / −)
   modèle au module        compter PII av./ap. → 0    pseudo vs anonyme
   remplacer par offsets   (email/tél/IBAN)           reflexion.md
```

| # | Étape | Geste | Fonction clé | Piège typique |
|---|---|---|---|---|
| 1 | **Charger** | colonne texte, gérer les `NaN` | `pd.read_csv()`, `.fillna("")` | `NaN` qui plante le `NLP(text)` |
| 2 | **Repérer** | regex (déterministe) + NER (probabiliste) | `re.compile`, `spacy.load`, `doc.ents` | Tout confier au NER (il ne voit ni email ni IBAN) |
| 3 | **Choisir** | une stratégie **justifiée** par le besoin | (décision — cf. table des stratégies) | Prendre Faker « par défaut » sans raison |
| 4 | **Anonymiser** | regex **avant** NER, modèle chargé au module | `RE.sub()`, remplacement par offsets | `text.replace(ent.text, …)` naïf (re-remplace des sous-chaînes) |
| 5 | **Vérifier** | compter les PII **avant/après** → *taux de fuite* (viser 0) | boucle + comptage résiduel | Déclarer « ça marche » sans chiffrer le résiduel |
| 6 | **Documenter** | faux positifs/négatifs + cadre légal | `reflexion.md` | Cacher les limites au lieu de les chiffrer |

---

### Les 4 invariants à mémoriser

1. **Regex AVANT NER.** Les PII de **forme fixe** (email, tél, IBAN) se
   détectent **exactement** par regex. Les retirer d'abord **réduit le bruit**
   envoyé au NER et lui évite un travail inutile (accessoirement, le NER peut
   découper un email en jetons). Le modèle probabiliste ne sert que pour le
   reste : noms, lieux.
2. **Le modèle NER est chargé UNE fois, au niveau module** (`NLP = spacy.load(...)`
   en haut du fichier), jamais dans la fonction — sinon ~1 s rechargé à chaque appel.
3. **On remplace par offsets** (`start_char`/`end_char`), **de la fin vers le
   début** du texte — un `text.replace(ent.text, …)` naïf re-remplacerait une
   sous-chaîne déjà masquée (et casse dès qu'un nom est inclus dans un autre).
4. **« Ça doit tourner sur un clone ».** Le modèle spaCy (`en_core_web_md`) et
   toute lib (Presidio…) **doivent** être dans `requirements.txt` + README. Code
   brillant non installable = zéro en prod.

---

## Le squelette minimal qui marche

```python
import re
import spacy

NLP = spacy.load("en_core_web_md")  # chargé UNE fois (invariant 2) — déclaré dans requirements

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(  # intl + parenthèses + extension (sinon on rate des numéros)
    r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3}[\s.-]?\d{3,4}(?:\s?x\d+)?")
IBAN_FULL_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")   # FR76...
IBAN_MASK_RE = re.compile(r"\*{2,}\d{4}")                         # ****3503

def anonymize_comments(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # [4a] Regex AVANT NER (invariant 1) — du plus spécifique au plus large
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = IBAN_FULL_RE.sub("[IBAN]", text)
    text = IBAN_MASK_RE.sub("[IBAN]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    # [4b] NER ensuite — remplacement PAR OFFSETS, de la fin vers le début (invariant 3)
    doc = NLP(text)
    ents = sorted((e for e in doc.ents if e.label_ in {"PERSON", "GPE", "LOC"}),
                  key=lambda e: e.start_char, reverse=True)  # ORG non masqué : cf. table
    for ent in ents:
        tag = "[NAME]" if ent.label_ == "PERSON" else "[LIEU]"
        # on découpe sur les positions, jamais text.replace() (sinon sous-chaînes)
        text = text[:ent.start_char] + tag + text[ent.end_char:]
    return text

df["manager_comments"] = df["manager_comments"].fillna("").apply(anonymize_comments)
df.head(200).to_csv("data/audit_sample_anonymized.csv", index=False)
```

> ⚠️ Ces regex couvrent les **cas courants**, pas tous (formats internationaux,
> nouveaux opérateurs…) — ce ne sont pas des « regex officielles ». Élargis-les
> au vu de **ton** corpus et mesure le taux de fuite résiduel.

---

### Choisir sa stratégie (étape 3)

| Stratégie | Sortie | Quand l'utiliser | Réversible ? |
|---|---|---|---|
| **Redaction** | `[NAME]`, `[EMAIL]` | défaut sobre, audit | Non (info perdue) |
| **Substitution (Faker)** | « Michael Wright » | le texte doit rester **lisible/naturel** | Non (sans table) |
| **Pseudonyme numéroté** | `[INDIVIDU_1]` (stable) | garder **qui parle de qui** | **Oui** *si* table conservée |
| **Hash salé** | `a3f9…` | recroiser/chaîner sans stocker l'identité en clair | **Non** (mais *chaînable*) |
| **Généralisation** | `[MANAGER]`/`[EMPLOYEE]` | préserver le **sens métier** | Non |

> ⚖️ **Pseudonymisation** (réversible : table conservée, *ou* hash d'un domaine
> restreint comme des noms → ré-identifiable par dictionnaire) → reste une
> donnée personnelle, **RGPD s'applique**. **Anonymisation** (irréversible) →
> sort du RGPD, mais l'absence de table est **nécessaire, pas suffisante** : il
> faut aussi qu'aucun résiduel ne ré-identifie (« le PDG de [ORG] »). Et on
> **perd** la capacité de recontacter / recroiser. C'est un arbitrage à
> défendre, pas un détail.

---

### Comment l'adapter

| Tu veux changer… | Tu modifies… |
|---|---|
| La langue du corpus | charge `fr_core_news_md` sur le FR (le modèle EN sur-masque `Le`, `un`, `RAS`) |
| La stratégie | l'étape [4b] — remplace `[NAME]` par un Faker, un pseudonyme stable, un hash |
| Les types de PII | ajoute une regex (n° sécu, plaque, matricule) **avant** le NER |
| L'outil de détection | Microsoft **Presidio** (`AnalyzerEngine`) — mais **déclare-le** dans requirements |
| Couvrir les PII **ambiguës** | un **LLM en appoint** *après* regex + NER — il **complète**, ne **remplace pas** le déterministe (pipeline hybride = norme émergente, cf. M7) |

---

### Quand ça ne marche pas — diagnostic rapide *(tiré des rendus M2-B2)*

| Symptôme | Cause probable | Vérifier |
|---|---|---|
| Des **numéros restent en clair** | `PHONE_RE` trop étroite (rate `+1`, `(…)`, `x5167`) | élargir la regex (indicatif + parenthèses + extension) |
| Un **IBAN `FR76…` passe** | regex ne couvre que l'IBAN **masqué** | ajouter `IBAN_FULL_RE` **avant** le partiel |
| `OSError E050: Can't find model` | modèle spaCy **non déclaré** / incohérent avec le repo | aligner sur `en_core_web_md` + README `python -m spacy download …` |
| `ModuleNotFoundError: presidio_analyzer` | dépendance **commentée** dans requirements | la décommenter — « si c'est pas dans requirements, ça n'existe pas » |
| Des **mots FR masqués** (`Le`, `un`, `RAS`) | modèle **EN** sur du **FR** | routage `fr_core_news_md`, ou documenter la limite |
| Un **acronyme métier détruit** (`PIP`, `RAS` → `[ORG]`) | masquage `ORG` aveugle | whitelist d'acronymes, ou ne masquer que `PERSON`/`GPE` |
| Une **sous-chaîne re-remplacée** par erreur | `text.replace(ent.text, …)` naïf | remplacer par offsets / du span le plus long d'abord |
| Un **nom avec `Jr.`/particule échappe** | `\b` casse l'appariement | élargir le motif **ou** assumer le faux négatif (documenté) |

> 💡 **Le test qui vaut tous les autres** : avant de rendre, lance ta fonction
> sur un commentaire qui contient **un vrai nom**. Si « John Smith » ressort en
> clair, le brief n'est pas fait — un email masqué ne suffit pas, c'est le **nom**
> qui identifie la personne.

---

*Fiche-pattern transverse — anonymisation de texte libre. Pour la classification
de texte, voir [`fiche-pattern-texte-nlp.md`](../05_NLP_Texte_PII/fiche-pattern-texte-nlp.md) ;
pour le stockage du texte (vectoriel/RAG), [`grille-decision-stockage.md`](../07_Data_Engineering/grille-decision-stockage.md).*
# Conventions de commit — Parcours IA ATOS

> Comment écrire des **messages de commit clairs** sur tes repos perso
> (M1-B1, M1-B2, et tous ceux qui suivent). Lecture : 5 min.

---

## Format

```
type(scope): description
```

- **type** : la nature du changement
- **scope** : la zone touchée
- **description** : courte phrase à l'impératif, ≤ 72 caractères, **en
  anglais** (convention pro), pas de point final

Exemple : `feat(api): add /predict endpoint with Pydantic validation`

---

## Les 6 types à connaître

| Type | Quand l'utiliser |
|---|---|
| `feat` | Nouvelle fonctionnalité (route, fonction, page) |
| `fix` | Correction d'un bug |
| `docs` | Modif de doc (README, commentaires, mini-cours) |
| `refactor` | Réorganisation du code sans changer son comportement |
| `test` | Ajout ou modif de tests |
| `chore` | Maintenance (gitignore, requirements, setup) |

---

## Scopes typiques sur tes briefs

| Scope | Ce qu'il couvre |
|---|---|
| `api` | Routes FastAPI, schémas Pydantic |
| `model` | Chargement, persistance, contract test du modèle |
| `tests` | Fichiers dans `tests/` |
| `docker` | Dockerfile, .dockerignore, docker-compose |
| `notebook` | Notebook EDA / entraînement |
| `eda` | Analyse exploratoire |
| `training` | Pipeline d'entraînement |
| `docs` | README, verdict.md, experiments.md |

Si le changement touche **plusieurs zones**, choisis le scope principal
ou crée plusieurs commits.

---

## Règles de la description

- ✅ **Impératif présent** : `add`, `fix`, `update`, `remove`
  (pas `added`, `adding`, `adds`)
- ✅ **Courte** : ≤ 72 caractères
- ✅ **Minuscule** au début (sauf nom propre, sigle)
- ✅ **Pas de point final**
- ✅ **En anglais** en général— convention dans le métier

---

## Exemples bien carrés ✅

```
feat(api): add /predict endpoint with Pydantic validation
feat(model): load joblib model in FastAPI lifespan
fix(api): handle missing categorical fields with 422 response
docs(readme): add quickstart in 3 commands
test(api): add 3 tests for /health, /predict ok and 422
refactor(training): extract preprocessing into src/preprocess.py
chore(docker): pin python:3.11-slim base image
```

---

## Anti-exemples ❌

| Mauvais commit | Pourquoi |
|---|---|
| `Update` | Aucune info |
| `fix bug` | Quel bug ? Quel scope ? |
| `WIP` | Pas de commits work-in-progress dans `main` |
| `feat: stuff` | Description vide de sens |
| `feat(api): added new endpoint.` | Participe passé + point final |
| `feat(api): I added a new endpoint to handle predictions for the model trained yesterday` | Trop long, met le détail en body si besoin |
| `feat: ajout endpoint` | Si tu choisis EN, reste en EN |

---

## Une seule intention par commit

Un commit = un changement cohérent. Si tu as fait 3 choses différentes,
fais 3 commits.

❌ `feat: add API, fix Dockerfile, update README`
✅ 3 commits séparés :

```
feat(api): add /predict endpoint
fix(docker): correct base image tag
docs(readme): update install instructions
```

---

## Workflow type

```bash
# Voir ce qui a changé
git status -s
git diff

# Stager les fichiers concernés (jamais git add . à l'aveugle)
git add app/main.py app/schemas.py

# Commit
git commit -m "feat(api): add /predict endpoint with Pydantic validation"

# Pousser
git push
```

Vérifie ton dernier commit avec `git log -1`.

---

## Pour aller plus loin (optionnel)

- [Conventional Commits — spec](https://www.conventionalcommits.org/fr/v1.0.0/)
- [Tim Pope — *A Note About Git Commit Messages*](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)

---

*Conventions de commit — ressource publique transverse, parcours IA ATOS.*
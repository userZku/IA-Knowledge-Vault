# MLflow — Tracking et Model Registry

> Thème : passer d'expériences locales à une gouvernance MLOps exploitable.
> Durée de lecture + pratique : ~40 min
> Pré-requis : [tracage experiments md](tracage-experiments-md.md), scikit-learn.

## Pourquoi cette fiche ?

`experiments.md` est parfait pour débuter. Dès qu'on multiplie les runs,
les jeux de données, et les personnes, MLflow devient utile pour :

- tracer automatiquement params, métriques, artefacts ;
- comparer visuellement les runs ;
- versionner et promouvoir des modèles (Staging, Production).

## Architecture minimale

1. Tracking server MLflow
2. Backend store (sqlite au début, postgres ensuite)
3. Artifact store (local, S3, MinIO)
4. Model Registry pour les versions de modèle

## Exemple minimal

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

mlflow.set_experiment("credit-risk")

with mlflow.start_run(run_name="rf_baseline"):
    params = {"n_estimators": 300, "max_depth": 12, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    f1 = f1_score(y_test, pred, average="macro")
    mlflow.log_metric("f1_macro", f1)

    mlflow.sklearn.log_model(model, artifact_path="model")
```

## Pattern de décision (simple)

- Run retenu si `f1_macro` >= baseline + contraintes métier.
- Enregistrer le modèle dans Registry avec une description claire.
- Promouvoir en `Staging`, puis `Production` après validation QA.

## Pièges fréquents

| Piège | Impact |
|---|---|
| Lancer des runs sans nom explicite | Historique illisible |
| Changer de split entre runs comparés | Conclusions fausses |
| Aucune stratégie de tags | Recherche impossible |
| Registry sans processus de promotion | "Production" arbitraire |

## Bonnes pratiques

- Tagger chaque run (`dataset_version`, `owner`, `ticket`).
- Logguer aussi temps d'entraînement et taille du modèle.
- Conserver une métrique métier (ex: recall classe défaut).
- Garder un changelog court dans la description des model versions.

## Checklist rapide

- [ ] Expérience MLflow créée
- [ ] Params + métriques logguées
- [ ] Artefact modèle enregistré
- [ ] Version de modèle créée en Registry
- [ ] Règle de promotion explicite (Staging -> Production)

## Pour aller plus loin

- Tracking : <https://mlflow.org/docs/latest/tracking.html>
- Model Registry : <https://mlflow.org/docs/latest/model-registry.html>

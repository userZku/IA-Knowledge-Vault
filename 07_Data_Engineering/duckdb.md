# DuckDB — SQL analytique local sur fichiers

> Thème : analyser rapidement CSV/Parquet sans déployer un SGBD.
> Durée de lecture + pratique : ~25 min
> Pré-requis : [parquet pyarrow](parquet-pyarrow.md), SQL de base.

## Pourquoi cette fiche ?

DuckDB est un moteur SQL analytique embarqué. Il lit directement les fichiers
Parquet/CSV, idéal pour:

- exploration rapide de data lake local ;
- agrégations lourdes sur laptop ;
- préparation de features avant entraînement.

## Cas d'usage typiques IA

- Audit de qualité sur plusieurs partitions Parquet.
- Jointures entre données brutes et référentiel métier.
- Échantillonnage reproductible pour entraînement.

## Exemple minimal (Python)

```python
import duckdb

con = duckdb.connect("analytics.duckdb")

query = """
SELECT region, date_trunc('month', event_time) AS month, COUNT(*) AS n_events
FROM read_parquet('data/events/*.parquet')
GROUP BY 1, 2
ORDER BY 2, 1
"""

df = con.execute(query).df()
print(df.head())
```

## Bonnes pratiques

- Stocker les datasets analytiques en Parquet (compression + schéma stable).
- Versionner les requêtes SQL importantes (dossier `sql/`).
- Préférer des vues/materialisations nommées pour reuse équipe.
- Toujours documenter les hypothèses (filtres, périodes, exclusions).

## Pièges fréquents

| Piège | Impact |
|---|---|
| Lire des CSV hétérogènes sans schéma explicite | Types incohérents |
| Mettre toute la logique dans un notebook unique | Reproductibilité faible |
| Oublier timezone sur colonnes date | Agrégations erronées |
| Ignorer les nulls dans les features dérivées | Drift silencieux |

## Checklist rapide

- [ ] Données principales en Parquet
- [ ] Requêtes critiques sauvegardées et commentées
- [ ] Vérification des types date/timestamp
- [ ] Exports de features tracés (date + source)

## Pour aller plus loin

- Doc officielle : <https://duckdb.org/docs/>
- SQL on files : <https://duckdb.org/docs/data/overview>

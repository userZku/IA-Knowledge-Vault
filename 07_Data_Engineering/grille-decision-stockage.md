# Grille de décision — Stockage & échelle des données

> **Compagnon de la [grille de décision C4](../10_Architecture_Cadrage_Produit/grille-decision-c4-reference.md)** (qui, elle,
> traite le choix du *modèle*). Ici on traite la décision **en amont** : où et
> comment **stocker** les données qui alimentent une solution IA, et **à quelle
> échelle** un outillage lourd devient justifié.
>
> Mobilisée dès **M3** (pipeline multi-sources Acerox), réutilisée en **M5**
> (déploiement / orchestration) et dans le **notebook certif M9**.
>
> **Statut** : version initiale, à enrichir en cours de promo.

---

## 🎯 À quoi sert cette grille ?

Beaucoup d'entre vous viennent du monde IT (DBA, intégrateur, archi) et
**connaissent déjà** OLTP, ETL, entrepôt de données. L'objectif n'est pas de
réapprendre ça — c'est de savoir **où ces briques s'insèrent dans une chaîne
IA**, et de poser le bon curseur :

> **Votre rôle ici = intégrateur de la donnée pour l'IA, pas data engineer
> à l'échelle.** Choisir le bon stockage est de votre ressort ; monter un
> cluster Spark ou un Snowflake, c'est une spécialité — vous devez savoir
> *quand* ça devient nécessaire, pas le mettre en œuvre dans ce parcours.

Quatre axes de décision :

1. **Nature de la donnée** (structurée / semi-structurée / non structurée)
2. **Motif d'accès** (transactionnel OLTP ↔ analytique OLAP ↔ similarité)
3. **Volume & fraîcheur** (taille, débit, fréquence de mise à jour)
4. **Échelle** (à partir de quand l'outil simple ne suffit plus)

…plus un critère transverse qui tranche les cas limites : le **coût** (cf. règle
d'or sobriété, axe 4).

> 🌳 **Arbre express (≈ 80 % des cas)** : *Données structurées ?* → **SQL**
> (SQLite → PostgreSQL). *Sinon, je veux chercher par le sens ?* → **vectoriel**
> (ChromaDB / pgvector). *Sinon* (images, binaires, archives) → **object store /
> fichiers**. Les quatre axes ci-dessous affinent ce premier tri.

---

## 📊 Axe 1 — Nature de la donnée → type de stockage

| Nature | Exemple parcours | Stockage adapté | Stockage à éviter |
|---|---|---|---|
| **Structurée** (lignes/colonnes, schéma stable) | Capteurs IoT, ERP Acerox | **Relationnel** (SQLite → PostgreSQL) | Vectoriel (inutile) |
| **Semi-structurée** (JSON imbriqué, schéma souple) | Export ERP JSON, logs applicatifs | Relationnel + colonnes JSON, ou **document** (MongoDB) | DWH rigide |
| **Non structurée — texte** | Rapports techniques, tickets, mails | **Vectoriel** (ChromaDB + embeddings) pour la recherche sémantique | Relationnel seul |
| **Non structurée — binaire** (images, audio) | Images PCB (M4-B2) | **Object store** (fichiers + chemins en base) ou Parquet | Blob en base relationnelle |
| **Append-only / immuable** (événements horodatés) | Logs machines, mesures continues | **Fichier colonne** (Parquet, append) ou time-series DB | Réécriture en place |

> **Réflexe M3** : une même mission mélange souvent **plusieurs natures** →
> **plusieurs stockages** (une *architecture polyglotte*). Acerox = relationnel
> (capteurs/ERP) **+** vectoriel (rapports). Ce n'est pas « choisir un stockage »
> mais **router chaque source vers le bon**.
>
> ⚠️ **Le vectoriel dépend de l'usage, pas de la nature seule** : un gros corpus
> de texte ne devient vectoriel **que si on veut le fouiller par le sens**
> (recherche sémantique / RAG). Pour juste le stocker ou le filtrer par
> mots-clés, des **fichiers**, **PostgreSQL** ou un **moteur de recherche**
> (Elasticsearch) suffisent. → c'est l'**axe 2** (motif d'accès) qui décide.

---

## 📊 Axe 2 — Motif d'accès → famille de système

| Motif d'accès | Question | Famille |
|---|---|---|
| **Transactionnel (OLTP)** | écritures fréquentes, lectures ciblées, intégrité forte ? | Base relationnelle (PostgreSQL), document (MongoDB) |
| **Analytique (OLAP)** | gros agrégats, jointures larges, lecture massive, peu d'écriture ? | **Entrepôt** (BigQuery, Snowflake, Redshift) / **data lake** (Parquet sur S3) |
| **Similarité sémantique** | « trouve-moi les passages proches de cette question » ? | **Base vectorielle** (ChromaDB, FAISS, pgvector) |
| **Diffusion / file** | flux d'événements à consommer en continu ? | Kafka / file de messages (hors scope parcours) |

> 📖 **OLTP vs OLAP** (ce n'est *pas* « OLAP = SQL ») : l'**OLTP** manipule
> **quelques lignes à la fois** (créer/lire/modifier un enregistrement, intégrité
> forte) ; l'**OLAP** répond à des questions qui **parcourent une grande partie
> des données** (agrégats, tableaux de bord, analyses historiques). C'est une
> affaire de **motif d'accès**, pas de langage.
>
> En clair : **OLTP = faire tourner le métier** ; **OLAP/DWH/lake = analyser à
> grande échelle** ; **vectoriel = retrouver par le sens** (la brique RAG vue
> en M3-B1 / M7-B2). Dans ce parcours, vos pipelines alimentent surtout de
> l'OLTP léger (SQLite/Postgres) ; le DWH est une **culture**, pas un livrable.
>
> 🔭 **Où vit le vectoriel dans une chaîne RAG** (mission étoile M3-B1, cœur M7) :
> ```
> données → préparation → base vectorielle (Chroma/pgvector) → récupération [→ LLM]
> ```
> Le futur LLM **ne lira pas** PostgreSQL directement : il consommera les passages
> **récupérés par similarité**. La base vectorielle est l'**index de recherche**,
> pas la base métier. *En M3 on s'arrête à la **récupération** ; la génération
> LLM vient en M7.*

---

## 📊 Axe 3 — Volume & fraîcheur

| Volume | Fraîcheur | Outil suffisant |
|---|---|---|
| **< 1 Go**, tient en RAM | batch quotidien | **pandas + SQLite**, fichier Parquet |
| **1–50 Go** | batch / horaire | **PostgreSQL**, Parquet partitionné, DuckDB |
| **50 Go – qq To** | analytique récurrente | **Entrepôt** (BigQuery/Snowflake) ou data lake |
| **> qTo / streaming** | temps réel | Calcul distribué (Spark/Dask), time-series DB, streaming |

> ⚠️ Ces seuils en Go sont des **ordres de grandeur**, pas des règles : un laptop
> 64 Go de RAM + **DuckDB** repousse la limite très loin. Le vrai déclencheur
> n'est pas un nombre de Go mais une **limite constatée** — *ça tient en RAM* →
> pandas ; *ça déborde* → DuckDB / PostgreSQL ; *une seule machine ne suffit
> plus* → Spark. (Même logique de « mur mesuré » que l'axe 4.)

---

## 🪜 Axe 4 — L'échelle : quand l'outil simple ne suffit plus

L'erreur classique = dégainer l'outil lourd « pour faire bien ». Le bon réflexe
= **monter d'un cran seulement quand on bute sur un mur mesuré.**

```
pandas (en RAM)
   │  ça déborde la RAM / c'est lent ?
   ▼
SQLite (OLTP local) / DuckDB (OLAP local) / Parquet   ← 95 % des cas s'arrêtent ici
   │  besoin multi-utilisateurs / réseau / intégrité ?
   ▼
PostgreSQL
   │  analytique lourde, To, plusieurs équipes ?
   ▼
Entrepôt (BigQuery/Snowflake) / data lake (Parquet + Spark/Dask)
   │  vraiment distribué, > To, traitements massivement parallèles ?
   ▼
Spark / cluster
```

| Outil | Quand c'est justifié | Quand c'est du sur-engineering |
|---|---|---|
| **Spark / Dask** | données qui **ne tiennent pas sur une machine**, calcul distribué | dataset de 50 k lignes (= 99 % de nos briefs) → **non** |
| **Entrepôt (DWH)** | analytique récurrente multi-équipes sur des To | une table de faits unique en projet de formation → **non** |
| **ETL orchestré** (Airflow/Prefect) | dizaines de jobs interdépendants, planifiés, monitorés | un script d'ingestion → cron + script suffit (vu en **M5**) |

> **Règle d'or sobriété (cf. principes ATOS)** : le bon stockage est **le plus
> simple — et le moins coûteux — qui répond au besoin réel**. On justifie une
> montée en échelle par un **mur constaté** (RAM, latence, concurrence), jamais
> par anticipation ou par mode. *« BigQuery, c'est possible, mais trop cher pour
> 5 Go ; Snowflake, excellent, mais inutile ici. »* Le **coût** (€ + maintenance)
> est un axe de décision à part entière.

---

## 💼 Cas types du parcours

### Cas M3 (Acerox — pipeline multi-sources industrielle)

- **Capteurs IoT** (CSV, ~50 k lignes, structuré, append) → **SQLite**
  relationnel (Postgres en prod). Pandas suffit pour l'ingestion.
- **ERP** (JSON, semi-structuré, semi-personnel) → relationnel après
  normalisation (+ pseudonymisation `ouvrier_id`).
- **Rapports techniques** (`.md`, non structuré) → **ChromaDB vectoriel** pour
  la recherche sémantique (mission étoile RAG).
- **Verdict** : pas *un* stockage mais **un routage par nature** ; Spark/DWH =
  hors-sujet à ce volume.

### Cas M5 (déploiement / orchestration) — *à compléter en M5*

- Bascule SQLite → PostgreSQL multi-env ; orchestration **cron + script**
  (Prefect optionnel). C'est là, et pas avant, qu'on parle ETL planifié.

---

## 🧠 Méthode d'utilisation

Pour chaque source d'un nouveau besoin :

1. **Qualifie la nature** (structurée / semi / texte / binaire / append)
2. **Identifie le motif d'accès** (OLTP / OLAP / similarité)
3. **Estime volume & fraîcheur**
4. **Choisis le stockage le plus simple** qui coche les 3 premiers points
5. **Ne monte en échelle que sur un mur mesuré** — et documente-le
6. **Trace la décision** dans `decisions.md` (M3) ou le notebook (M9)

---

## 📚 Pour aller plus loin

- **Martin Kleppmann — *Designing Data-Intensive Applications*** : la référence
  sur OLTP/OLAP, stockage, partitionnement (culture archi)
- **DuckDB** : <https://duckdb.org/> — « le SQLite de l'analytique », souvent
  la bonne réponse avant de penser DWH
- **ChromaDB** : <https://docs.trychroma.com/> — base vectorielle locale (RAG M3/M7)
- **pgvector** : <https://github.com/pgvector/pgvector> — vectoriel **dans**
  PostgreSQL, quand on ne veut pas une base de plus

---

*Grille pédagogique ATOS — Stockage & échelle. Version v1.1 par
Marianne Arrué (formatrice), 2026-06-28. Compagnon de la grille de décision C4.*
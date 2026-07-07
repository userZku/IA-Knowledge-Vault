# Gradio — Prototyper des interfaces IA en minutes

> Thème : démo rapide d'un modèle ML/NLP sans front-end complet.
> Durée de lecture + pratique : ~20 min
> Pré-requis : Python, endpoint de prédiction ou fonction Python modèle.

## Pourquoi cette fiche ?

Quand l'objectif est de valider un usage avec des métiers, Gradio permet de
livrer une UI testable très vite, avec moins de friction que du front custom.

## Quand choisir Gradio plutôt que Streamlit ?

- Gradio : formulaires IA, démos modèles, partage rapide, HF Spaces.
- Streamlit : dashboards plus riches, navigation multi-pages, data app interne.

Les deux sont compatibles avec un backend FastAPI.

## Exemple minimal

```python
import gradio as gr

def score_sentiment(text: str) -> str:
    text = text.lower()
    if "excellent" in text or "parfait" in text:
        return "positif"
    if "horrible" in text or "nul" in text:
        return "negatif"
    return "neutre"

demo = gr.Interface(
    fn=score_sentiment,
    inputs=gr.Textbox(lines=4, label="Avis client"),
    outputs=gr.Label(label="Sentiment"),
    title="Démo sentiment FR",
    description="Prototype rapide pour validation métier",
)

if __name__ == "__main__":
    demo.launch()
```

## Bonnes pratiques

- Afficher clairement que c'est un prototype (pas une prod).
- Préparer 5-10 exemples métier dans l'interface pour la démo.
- Logger les retours utilisateurs (faux positifs/faux négatifs).
- Garder la même nomenclature de classes que l'API cible.

## Pièges fréquents

| Piège | Impact |
|---|---|
| Confondre prototype et service production | Failles de sécurité/perf |
| Aucun timeout sur appel backend | Interface qui bloque |
| Labels non alignés avec l'API | Incompréhension métier |
| Aucun jeu d'exemples de test | Demo peu convaincante |

## Checklist rapide

- [ ] Interface lancée localement
- [ ] Exemples métier préchargés
- [ ] Réponses lisibles pour non-technique
- [ ] Limites du prototype explicitées

## Pour aller plus loin

- Doc officielle : <https://www.gradio.app/docs>
- Hugging Face Spaces : <https://huggingface.co/spaces>

# FastAPI + JWT — Authentification API

> Thème : sécuriser une API ML exposée à des clients internes/externes.
> Durée de lecture + pratique : ~35 min
> Pré-requis : [fastapi](fastapi.md), HTTP de base.

## Pourquoi cette fiche ?

Une API de prédiction sans authentification finit souvent exposée "en clair"
sur un réseau interne, puis réutilisée hors cadre. Le JWT permet :

- d'identifier qui appelle l'API ;
- de limiter l'accès à certains endpoints ;
- d'auditer les usages (qui, quand, quoi).

Sur un parcours IA, c'est un standard minimal avant mise en preprod/prod.

## Concepts clés

- JWT = token signé, transporté dans `Authorization: Bearer <token>`.
- Le token contient des claims (ex: `sub`, `role`, `exp`).
- Signature valide != droits suffisants : il faut vérifier les rôles.
- Access token court (ex: 15 min) + refresh token optionnel.

## Pattern recommandé (MVP propre)

1. Endpoint `/auth/token` qui vérifie un utilisateur et émet un JWT.
2. Dépendances FastAPI pour décoder/valider le token.
3. Vérification du rôle sur les routes sensibles (`/predict`, `/admin/...`).
4. Rotation de secret via variable d'environnement.

```python
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

SECRET_KEY = "change-me-in-env"
ALGO = "HS256"

def create_access_token(sub: str, role: str, minutes: int = 15) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGO)

def current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        claims = jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
        if "sub" not in claims:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return claims
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide") from exc

@app.get("/predict")
def predict(user=Depends(current_user)):
    return {"ok": True, "caller": user["sub"], "role": user.get("role", "unknown")}
```

## Bonnes pratiques

- Stocker `SECRET_KEY` dans une variable d'env, jamais en dur dans Git.
- Ajouter `iss` et `aud` pour éviter des tokens réutilisés hors contexte.
- Vérifier `exp` et gérer un léger `clock skew` côté serveur.
- Journaliser les refus d'accès sans logguer le token brut.

## Pièges fréquents

| Piège | Impact |
|---|---|
| Secret hardcodé dans le code | Fuite de credentials via repo |
| Token très long (24h+) | Fenêtre d'attaque trop grande |
| Vérification signature seulement | Contournement de l'autorisation métier |
| Aucune gestion de role | Tous les utilisateurs voient tout |

## Checklist rapide

- [ ] Endpoint `/auth/token` actif
- [ ] `Bearer` requis sur `/predict`
- [ ] Rôles vérifiés sur endpoints sensibles
- [ ] Secret configuré via environnement
- [ ] Tests 401/403 couvrent les cas invalides

## Pour aller plus loin

- FastAPI security : <https://fastapi.tiangolo.com/tutorial/security/>
- OAuth2 RFC : <https://www.rfc-editor.org/rfc/rfc6749>
- JWT RFC : <https://www.rfc-editor.org/rfc/rfc7519>

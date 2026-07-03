# Despliegue en Azure via GitHub Actions

Cada push a `main` dispara el workflow automáticamente:
build Docker → push a ACR → deploy en Azure Container Apps.

---

## Configuración inicial (solo una vez)

### 1. Crear los recursos en Azure

```bash
az login

# Resource group
az group create --name fondos-ai-rg --location westeurope

# Azure Container Registry
az acr create \
  --name fondosairegistry \
  --resource-group fondos-ai-rg \
  --sku Basic \
  --admin-enabled true

# Azure Container Apps environment
az containerapp env create \
  --name fondos-ai-env \
  --resource-group fondos-ai-rg \
  --location westeurope

# Container App (primera vez, con imagen placeholder)
az containerapp create \
  --name fondos-ai \
  --resource-group fondos-ai-rg \
  --environment fondos-ai-env \
  --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
  --target-port 8501 \
  --ingress external \
  --min-replicas 1
```

### 2. Obtener las credenciales para GitHub Secrets

```bash
# Service principal para que GitHub pueda autenticarse en Azure
az ad sp create-for-rbac \
  --name "fondos-ai-github" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/fondos-ai-rg \
  --sdk-auth
```

Copia el JSON completo que devuelve — lo necesitas para el secret `AZURE_CREDENTIALS`.

```bash
# Credenciales del Container Registry
az acr credential show --name fondosairegistry
```

### 3. Añadir los Secrets en GitHub

Ve a tu repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Valor |
|---|---|
| `AZURE_CREDENTIALS` | JSON completo del `az ad sp create-for-rbac` |
| `ACR_LOGIN_SERVER` | `fondosairegistry.azurecr.io` |
| `ACR_NAME` | `fondosairegistry` |
| `ACR_USERNAME` | username del `az acr credential show` |
| `ACR_PASSWORD` | password del `az acr credential show` |
| `GEMINI_API_KEY` | tu clave de la API de Gemini |

---

## Desplegar

Con todo configurado, solo tienes que hacer push a `main`:

```bash
git add .
git commit -m "deploy"
git push origin main
```

GitHub Actions construye la imagen, la sube a ACR y actualiza la Container App. En ~5 minutos la app está actualizada.

---

## Obtener la URL pública

```bash
az containerapp show \
  --name fondos-ai \
  --resource-group fondos-ai-rg \
  --query properties.configuration.ingress.fqdn \
  --output tsv
```

Formato: `https://fondos-ai.XXXX.westeurope.azurecontainerapps.io`

---

## Probar localmente antes de hacer push

```bash
pip install streamlit
streamlit run app.py
# → http://localhost:8501
```

O con Docker:

```bash
docker build -t fondos-ai .
docker run -p 8501:8501 -e GEMINI_API_KEY=tu_clave fondos-ai
# → http://localhost:8501
```

---

## Borrar todo cuando ya no lo necesites

```bash
az group delete --name fondos-ai-rg --yes
```

# Deploying to homelab k8s

Bot needs only outbound network (it long-polls Telegram). No Ingress, no LoadBalancer, no Tailscale exposure.

## One-time setup

### 1. Create the namespace and the secret manually

The secret stays OUT of git — ArgoCD won't manage it, so updates won't clobber it.

```sh
kubectl create namespace language-tutor

kubectl create secret generic language-tutor-secrets \
  --namespace language-tutor \
  --from-literal=TELEGRAM_BOT_TOKEN='<paste from BotFather>' \
  --from-literal=ANTHROPIC_API_KEY='<paste from console.anthropic.com>'
```

### 2. Build and push the image

Pick a registry. Two easy options:

**Option A — GHCR (requires GitHub repo + auth):**

```sh
docker build -t ghcr.io/<your-user>/language-tutor:v0.1.0 .
echo $GITHUB_PAT | docker login ghcr.io -u <your-user> --password-stdin
docker push ghcr.io/<your-user>/language-tutor:v0.1.0
```

**Option B — local registry on the homelab:**

```sh
docker build -t registry.homelab.local/language-tutor:v0.1.0 .
docker push registry.homelab.local/language-tutor:v0.1.0
```

Then edit `deployment.yaml` and replace the `image:` field with what you actually pushed.

### 3. Register the ArgoCD Application

Copy `argocd-application.yaml` into your `homelab-k8s` repo (e.g. `apps/workloads/language-tutor.yaml`),
edit the `repoURL` to point at this repo's git URL, and commit. ArgoCD will sync the rest automatically.

## Updating

```sh
# 1. Make your code change, bump the version tag
docker build -t <registry>/<user>/language-tutor:v0.2.0 .
docker push <registry>/<user>/language-tutor:v0.2.0

# 2. Update deployment.yaml with the new tag, commit, push
# 3. ArgoCD syncs the new image. Pod restarts. PVC stays intact → all user data preserved.
```

## Data persistence

The SQLite DB lives on a 1Gi PVC mounted at `/data`. It survives:

- Pod restarts
- Image updates
- Rolling deploys

It does NOT survive:

- Deleting the PVC manually
- Tearing down the cluster

## Verifying after deploy

```sh
kubectl -n language-tutor get pods
kubectl -n language-tutor logs -f deploy/language-tutor
```

You should see:

```
bot starting (polling) — model=claude-haiku-4-5 db=/data/tutor.db curriculum=83 lessons
```

Then DM the bot in Telegram — it should respond.

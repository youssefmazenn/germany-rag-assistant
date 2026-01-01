# -----------------------------
# Project configuration
# -----------------------------
IMAGE_NAME = germany-rag-assistant
PORT = 8000
CHROMA_DIR = $(PWD)/chroma_db

K8S_NAMESPACE := germany-rag
K8S_DIR := infra/k8s
APP_NAME := rag-api
SERVICE_NAME := rag-service

# -----------------------------
# Docker commands
# -----------------------------

docker-build:
	docker build -f infra/docker/Dockerfile -t $(IMAGE_NAME):latest .

docker-run:
	docker run --rm -p $(PORT):8000 \
		--env-file .env \
		-v "$(CHROMA_DIR):/app/chroma_db" \
		$(IMAGE_NAME):latest

docker-stop:
	docker ps -q --filter ancestor=$(IMAGE_NAME):latest | xargs -r docker stop

# -----------------------------
# Local development (optional)
# -----------------------------

run-local:
	python -m uvicorn app.main:app --reload


k8s-apply:
	kubectl apply -f $(K8S_DIR)/namespace.yaml
	kubectl apply -f $(K8S_DIR)/configmap.yaml
	kubectl apply -f $(K8S_DIR)/pvc.yaml
	kubectl apply -f $(K8S_DIR)/deployment.yaml
	kubectl apply -f $(K8S_DIR)/service.yaml

k8s-restart:
	kubectl -n $(K8S_NAMESPACE) rollout restart deployment $(APP_NAME)

k8s-pods:
	kubectl -n $(K8S_NAMESPACE) get pods -w

k8s-logs:
	kubectl -n $(K8S_NAMESPACE) logs deploy/$(APP_NAME) --tail=50

k8s-forward:
	kubectl -n $(K8S_NAMESPACE) port-forward svc/$(SERVICE_NAME) 8000:8000

k8s-delete:
	kubectl delete namespace $(K8S_NAMESPACE)


# -----------------------------
# Project configuration
# -----------------------------
IMAGE_NAME = germany-rag-assistant
PORT = 8000
CHROMA_DIR = $(PWD)/chroma_db

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
# Mongo support
#
# db.book.find({id: 300})
#
# db.book.aggregate([
# 	{
# 		"$vectorSearch": {
# 			"index": "vector_index",
# 			"exact": true,
# 			"path": "embedding",
# 			"queryVector": [1.2, 2.2, 3.2, 4.2],
# 			"limit": 10
# 		}
# 	},
# 	{
# 		"$project": {
# 			"text": 1,
# 			"embedding": 1,
# 			"score": {
# 				"$meta": "vectorSearchScore"
# 			}
# 		}
# 	}
# ])

# ==============================================================================
# Install dependencies

install:
	brew install mongosh
	brew install ollama
	brew install mplayer
	brew install pgcli
	brew install uv

docker:
	docker pull mongodb/mongodb-atlas-local
	docker pull ghcr.io/open-webui/open-webui:v0.6.18
	docker pull postgres:17.5

ollama-pull:
	ollama pull bge-m3:latest
	ollama pull qwen2.5vl:latest
	ollama pull gpt-oss:latest
	ollama pull hf.co/gpustack/bge-reranker-v2-m3-GGUF:Q8_0

python-install:
	rm -rf .venv
	uv venv --python 3.12 && uv lock && uv sync

# ==============================================================================
# Ollama Settings

OLLAMA_CONTEXT_LENGTH := 65536
OLLAMA_NUM_PARALLEL := 4
OLLAMA_MAX_LOADED_MODELS := 2

# ==============================================================================
# Examples

example01:
	go run cmd/examples/example01/main.go
	@@ -99,7 +48,7 @@ example07:
	go run cmd/examples/example07/main.go

example08:
	go run cmd/examples/example08/*.go

example09-step1:
	go run cmd/examples/example09/step1/main.go
	@@ -132,29 +81,44 @@ example10-step4:
	export OLLAMA_CONTEXT_LENGTH=$(OLLAMA_CONTEXT_LENGTH) && \
	go run cmd/examples/example10/step4/*.go

example10-step5:
	export OLLAMA_CONTEXT_LENGTH=$(OLLAMA_CONTEXT_LENGTH) && \
	go run cmd/examples/example10/step5/*.go

example11-step1:
	go run cmd/examples/example11/step1/main.go

example11-step2:
	export OLLAMA_CONTEXT_LENGTH=$(OLLAMA_CONTEXT_LENGTH) && \
	go run cmd/examples/example11/step2/*.go

example12-step1:
	go run cmd/examples/example12/step1/main.go

talk:
	export OLLAMA_CONTEXT_LENGTH=$(OLLAMA_CONTEXT_LENGTH) && \
	go run cmd/talk/main.go

# ==============================================================================
# Manage project

compose-up:
	rm -rf zarf/docker/db_data && \
	mkdir -p zarf/docker/db_data/db zarf/docker/db_data/configdb && \
	chmod -R 777 zarf/docker/db_data && \
	docker compose -f zarf/docker/compose.yaml up

compose-down:
	@@ -167,10 +131,7 @@ compose-logs:
# Ollama tooling

ollama-up:
	export OLLAMA_NUM_PARALLEL=$(OLLAMA_NUM_PARALLEL) && \
	export OLLAMA_MAX_LOADED_MODELS=$(OLLAMA_MAX_LOADED_MODELS) && \
	export OLLAMA_CONTEXT_LENGTH=$(OLLAMA_CONTEXT_LENGTH) && \
	ollama serve

ollama-logs:
	tail -f -n 100 ~/.ollama/logs/server.log

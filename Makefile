DOCKER_DIR:= ./docker
SRC_DIR:= ./src
DATASET_DIR:= ./datasets
MODELS_DIR:= ./models/nerf
SRC_FILE := $(wildcard	$(SRC_DIR)/*.py  $(SRC_DIR)/$(DATASET_DIR)/*.py $(SRC_DIR)/$(MODELS_DIR)/*.py)

.PHONY: docker-run
docker-run:
	-@sh $(DOCKER_DIR)/docker_run.sh
    
.PHONY: format
format:
	@echo "Format: " $(SRC_FILE)
	yapf -i  $(SRC_FILE)

.PHONY: clean
clean:
	-@sudo rm -rvf ./exp
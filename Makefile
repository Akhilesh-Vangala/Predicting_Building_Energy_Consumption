CONFIG ?= configs/default.yaml
PY     ?= python

.PHONY: all features train ablation cluster failure clean test

all: features train ablation cluster failure

features:
	$(PY) -m scripts.build_features --config $(CONFIG)

train:
	$(PY) -m scripts.train_all --config $(CONFIG)

ablation:
	$(PY) -m scripts.run_ablation --config $(CONFIG)

cluster:
	$(PY) -m scripts.run_clustering --config $(CONFIG)

failure:
	$(PY) -m scripts.run_failure_analysis --config $(CONFIG)

test:
	$(PY) -m pytest tests -v

clean:
	rm -rf results/metrics/* results/plots/* results/tables/*
	find . -type d -name __pycache__ -exec rm -rf {} +

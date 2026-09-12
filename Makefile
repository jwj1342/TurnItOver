.PHONY: bootstrap web typecheck test test-browser docs smoke preview clean

bootstrap:
	bash scripts/bootstrap_login.sh

web:
	cd web && npm run build

typecheck:
	cd web && npm run typecheck

test:
	pytest

test-browser:
	pytest -m browser

docs:
	python -m turnitover render-docs

smoke:
	python -m turnitover generate --config configs/generate_toy.yaml --n-samples 4 --out data/runs/smoke

preview:
	python -m turnitover preview

clean:
	rm -rf web/dist .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

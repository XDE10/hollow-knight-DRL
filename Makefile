# HK-RL developer task runner.
# Schema codegen feeds both Python and the C# mod from schema/hkrl.fbs.

PY        := python
FLATC     ?= flatc
FLATC_PY  ?= $(FLATC)
CSHARP_FLATC_VERSION ?= 23.5.26
CONDA_MOD_FLATC := $(shell \
	if command -v conda >/dev/null 2>&1 && \
	   conda env list 2>/dev/null | awk '{print $$1}' | grep -qx "hkrl-mod-build"; then \
		conda run -n hkrl-mod-build which flatc 2>/dev/null | tail -n 1; \
	fi)
FLATC_CS  ?= $(if $(CONDA_MOD_FLATC),$(CONDA_MOD_FLATC),$(FLATC))
PKG_DIR   := python
FBS       := schema/hkrl.fbs
PY_SCHEMA := python/hkrl/schema
CS_SCHEMA := mod/HKRLEnvMod/Schema

.DEFAULT_GOAL := help
.PHONY: help gen-schema gen-schema-py gen-schema-cs install check lint format-check typecheck test fmt clean

help: ## Show available make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- Schema codegen (single source of truth -> C# + Python) ----------------
gen-schema: gen-schema-py gen-schema-cs ## Regenerate all FlatBuffers bindings

gen-schema-py: ## Generate Python FlatBuffers bindings
	mkdir -p $(PY_SCHEMA)
	rm -rf $(PY_SCHEMA)/HKRL
	$(FLATC_PY) --python -o $(PY_SCHEMA) $(FBS)

gen-schema-cs: ## Generate C# FlatBuffers bindings
	@version="$$($(FLATC_CS) --version)"; \
	case "$$version" in \
	  *"$(CSHARP_FLATC_VERSION)"*) ;; \
	  *) echo "error: C# schema generation requires flatc $(CSHARP_FLATC_VERSION); got: $$version" >&2; \
	     echo "Use environment-mod-build.yml, or pass FLATC_CS=/path/to/flatc-$(CSHARP_FLATC_VERSION)." >&2; \
	     exit 2 ;; \
	esac
	mkdir -p $(CS_SCHEMA)
	rm -rf $(CS_SCHEMA)/HKRL
	$(FLATC_CS) --csharp -o $(CS_SCHEMA) $(FBS)

# ---- Python dev ------------------------------------------------------------
install: ## Editable install with dev extras
	pip install -e "$(PKG_DIR)[dev]"

check: gen-schema format-check lint typecheck test ## Run local quality gates

lint: ## ruff lint
	cd $(PKG_DIR) && ruff check .

format-check: ## ruff format check
	cd $(PKG_DIR) && ruff format --check .

fmt: ## ruff format
	cd $(PKG_DIR) && ruff format .

typecheck: ## mypy
	cd $(PKG_DIR) && mypy hkrl

test: ## pytest unit suite
	cd $(PKG_DIR) && pytest

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(PKG_DIR)/.ruff_cache $(PKG_DIR)/.mypy_cache $(PKG_DIR)/.pytest_cache

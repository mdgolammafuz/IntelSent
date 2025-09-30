.PHONY: test
test:
\tSKIP_CHAIN_INIT=1 pytest -q --maxfail=1 --disable-warnings

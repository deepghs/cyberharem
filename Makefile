.PHONY: webui

CONDA    ?= $(shell which conda)
BASH     ?= $(shell which bash)
WEBUI_SH     ?= $(shell readlink -f ${CH_WEBUI_DIR}/../webui.sh)
WEBUI_SH_DIR ?= $(shell readlink -f ${CH_WEBUI_DIR}/..)


webui:
	cd ${WEBUI_SH_DIR} && \
		${CONDA} run --live-stream --no-capture-output -n webui \
		${BASH} webui.sh -f --port ${CH_WEBUI_PORT} --api --share --max-batch-count 128 \
		--listen --no-gradio-queue --enable-insecure-extension-access --xformers --nowebui

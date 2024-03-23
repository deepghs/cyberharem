.PHONY: webui

CONDA    ?= $(shell which conda)

WEBUI_SH     ?= $(shell readlink -f ${CH_WEBUI_DIR}/../webui.sh)
WEBUI_SH_DIR ?= $(shell readlink -f ${CH_WEBUI_DIR}/..)


webui:
	cd ${WEBUI_SH_DIR}
	${CONDA} run --live-stream --no-capture-output -n ${CH_KOHYA_CONDA_ENV} cat webui.sh

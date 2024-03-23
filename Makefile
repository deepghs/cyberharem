.PHONY: webui

WEBUI_SH ?= $(shell readlink -f ${CH_WEBUI_DIR}/../webui.sh)

webui:
	echo ${WEBUI_SH}

# One-word entry points for the whole pipeline.  `make help` lists them.
CONFIG ?= configs/default.yaml
SET    ?=
PY     ?= python

.PHONY: help install data detect track eval ablation calibrate demo yolo-data reid-data test smoke lint

help:             ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-12s %s\n", $$1, $$2}'

install:          ## install everything (detector, ReID, data download, TrackEval, dev tools)
	pip install -e ".[vision,data,eval,dev]"

data:             ## download SportsMOT and keep only basketball (needs HF login + accepted licence)
	$(PY) scripts/download_data.py --splits train val

detect:           ## run YOLO once and cache detections (val split)
	hooptrack detect --config $(CONFIG) --set $(SET)

track:            ## track the val split using cached detections, then evaluate
	hooptrack track --config $(CONFIG) --set detector.type=cached $(SET) --eval

eval:             ## full evaluation incl. HOTA/AssA (TrackEval)
	hooptrack evaluate --config $(CONFIG) --hota --set $(SET)

ablation:         ## add one component at a time: motion -> GMC -> ReID -> memory -> GT boxes
	$(PY) scripts/run_ablation.py --config $(CONFIG) --hota

calibrate:        ## measure ReID separability on TRAIN and suggest thresholds
	$(PY) scripts/calibrate_reid.py --config $(CONFIG) --set $(SET)

demo:             ## render an annotated video for one validation clip
	hooptrack track --config $(CONFIG) --set detector.type=cached output.save_video=true \
		"data.only=[v_00HRwkvvjtQ_c001]" output.dir=outputs/runs/demo $(SET)

yolo-data:        ## build a single-class 'player' YOLO dataset from the train split
	$(PY) scripts/prepare_yolo_dataset.py --stride 5

reid-data:        ## export Market-1501-style crops to fine-tune the ReID model
	$(PY) scripts/export_reid_crops.py --stride 3

test:             ## unit tests
	pytest

smoke:            ## end-to-end run on a rendered synthetic clip (no data, no GPU)
	$(PY) scripts/smoke_test.py

lint:             ## static checks
	ruff check .

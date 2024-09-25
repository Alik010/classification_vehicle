PRIVATE_KEY_PATH = ~/.ssh/ssh_priv
USERNAME = alik

install:
	pip install -r requirements.txt

download_dataset:
	mkdir dataset_coco
	rsync -aSvuc -e 'ssh -p 22022 -i $(PRIVATE_KEY_PATH)' $(USERNAME)@ml-server.avtodoria.ru:/mnt/storage_hdd/dataset_storage/multilabel_street_light_cls/dataset_coco/ dataset_coco/

lint:
	PYTHONPATH=. flake8 src
	PYTHONPATH=. black src
	PYTHONPATH=. nbstripout notebooks/*.ipynb

clean-logs: ## Clean logs
	rm -rf logs/**

train: ## Train the model
	python src/train.py

test:
	python src/eval.py

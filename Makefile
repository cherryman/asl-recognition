.PHONY: help
help:
	@echo "Available targets:"
	@echo "  train    Train the model."
	@echo "  check    Test the model's accuracy."
	@echo ""
	@echo "Dependencies:"
	@echo "  kaggle-api"
	@echo "  p7zip"
	@echo "  rsync"

.PHONY: download extract merge train check
download: build/ build/data1.zip build/data2.zip build/data3.zip
extract: download build/data1 build/data2 build/data3
merge: data/
train: build/model.pth

check:
	./model.py test

build/model.pth: ./model.py
	./model.py train

build/data1.zip:
	kaggle datasets download grassknoted/asl-alphabet \
	&& mv asl-alphabet.zip $@

build/data1: build/data1.zip
	7z x $? -aoa -o$@
	touch $@

build/data2.zip:
	kaggle datasets download danrasband/asl-alphabet-test \
	&& mv asl-alphabet-test.zip $@

build/data2: build/data2.zip
	7z x $? -aoa -o$@
	touch $@

build/data3.zip:
	kaggle datasets download mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out \
	&& mv asl-rgb-depth-fingerspelling-spelling-it-out.zip $@

build/data3: build/data3.zip
	7z x $? -aoa -o$@
	touch $@

build/:
	mkdir -p $@

data: build/data1 build/data2 build/data3
	mkdir -p $@
	rm -r data/*
	rsync -am --include='[A-Z]/***' --exclude='*' build/data1/asl_alphabet_train/asl_alphabet_train/ $@/
	rsync -am --include='[A-Z]/***' --exclude='*' build/data2/ $@/
	for d in $@/[A-Z]; do mv $$d $$(echo $$d | tr '[A-Z]' '[a-z]'); done
	find build/data3/dataset5 -mindepth 1 -maxdepth 1 \
	  -exec rsync -am --include='color*.png' --include='*/' --exclude='*' '{}/' $@/ \; > /dev/null
	touch $@


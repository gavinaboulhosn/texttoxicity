.PHONY: test

env:
	sudo easy_install pip && \
	pip install virtualenv && \
	virtualenv env && \
	. env/bin/activate && \
	make deps

deps:
	pip install -r requirements.txt

clean:
	rm -fr build \
	rm -fr dist \
	find . -name '*.pyc' -exec rm -f {} \
	find . -name '*.pyo' -exec rm -f {} \
	find . -name '*~' -exec rm -f {}

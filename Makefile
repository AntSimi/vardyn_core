.PHONY: clean install develop docclean doc test check


clean:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf vardyn_core.egg-info
	rm -rf __pycache__
	find vardyn_core/ -name '*.nbi' -delete
	find vardyn_core/ -name '*.nbc' -delete
	find vardyn_core/ -name '*.so' -delete
	find vardyn_core/ -name '*.c' -delete
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete


install:
	pip install .

develop:
	pip install -e .

doc:
	export OMP_NUM_THREADS=1
	sphinx-build -M html doc/source build/sphinx

docclean:
	rm -rf doc/build
	rm -rf doc/source/_autosummary
	rm -rf doc/source/auto_examples/
	rm -rf doc/source/gen_modules
	rm -f doc/source/sg_execution_times.rst

	find doc/ -name '*.pyc' -delete


check:
	isort vardyn_core
	black vardyn_core
	flake8 vardyn_core --exit-zero
	isort examples
	black examples
	flake8 examples --exit-zero
	
build-go:
	cd go && go build -o ../py/gotorch/_gotorch.so

test-go:
	cd tests/go_tests && go test ./...

test-python:
	pytest tests/py_tests

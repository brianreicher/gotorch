from setuptools import setup, Extension

gotorch_extension = Extension(
    "gotorch._gotorch",
    sources=["../go/main.go"],
    extra_compile_args=["-shared"],
)

setup(
    name="gotorch",
    version="0.1.0",
    packages=["gotorch"],
    ext_modules=[gotorch_extension],
)

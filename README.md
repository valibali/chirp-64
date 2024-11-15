# mypackage

Boilerplate code for Poetry based Python package

## Installation

```bash
pip install mypackage \
    --index-url https://anu9rng:AP6eY5xuhS1MqAdy5jedftw3ndQq7MHjXL8Rpb@rb-artifactory.bosch.com/artifactory/api/pypi/python-virtual/simple \
    --extra-index-url https://anu9rng:AP6eY5xuhS1MqAdy5jedftw3ndQq7MHjXL8Rpb@rb-artifactory.bosch.com/artifactory/api/pypi/python-virtual/simple
```

## Development

- Requirements:
  - Python 3.8+
  - Poetry 1.8+
  - VSCode with devcontainers extension working

1. Clone this repository

2. [Initial setup instructions](#initial-setup-instructions) (only required for the first time working on a mypackage)

3. Navigate to the newly created repository

4. Open the repository in VSCode

    ```sh
      code .
    ```

5. Once asked, open the workspace in the devcontainer offered by VSCode. At first it's goint to build the container.

6. Initialize poetry environment if you didn't do it already for this repo (This is done automatically in VSCode when you open workspace)

    ```sh
    poetry install
    ```

7. Activate the virtual environment

    ```sh
    poetry shell
    ```

8. You can (optionally) run all the checks on the whole code as well using(i.e. formatting, linting, type checking, ...)

    ```sh
    pre-commit run -a
    ```

### Dependency management

It is our recommendation to [check in the poetry lockfile](https://python-poetry.org/docs/basic-usage/#committing-your-poetrylock-file-to-version-control).

- Install latest poetry dependencies versions

  ```sh
  poetry update
  ```

  The command updates all dependendencies as well as the [poetry lockfile](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock).

- Add new required packages/dependencies (that will be installed together with mypackage)

  Update `[tool.poetry.dependencies]` section in pyproject.toml, and run,

  ```sh
  poetry update
  ```

- Add new required packages/dependencies only for development (that will NOT be installed together with mypackage)

  Update `[tool.poetry.group.dev.dependencies]` section in pyproject.toml, and run,

  ```sh
  poetry update
  ```

### Testing

Run all tests:

```sh
pytest
```

To see the stdout/stderr for the tests:

```sh
pytest -s
```

We recommend to use the integrated VSCode python testing extension to run (debug) the tests.<br>
https://code.visualstudio.com/docs/python/testing#_configure-tests

### Documentation

Generate the documentation:

```sh
mkdocs build
```

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
of the public signatures of the source code.

The generated documentation can be found under /site.

### Releasing

- Build the package

  ```sh
  poetry build
  ```

  Creates the package under /dist.

- Publish the package

  ```sh
  poetry publish -r <repository_name>
  ```

  Publishing of packages should be done only by the CI pipeline.


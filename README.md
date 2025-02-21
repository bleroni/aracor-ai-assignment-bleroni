# aracor-ai-assignment-bleroni

## Running locally
`poetry install --with development`

## Executing individual files
`poetry run python -m src.models.openai_config`

## Running tests
`poetry add --dev pytest`  
`poetry run pytest -v`

## Running pylint
`poetry add --dev pylint`  
`poetry run pylint src/`

## Running black
`poetry add --dev black`  
`poetry run black .`

## Runing isort
`poetry add --dev isort`  
`poetry run isort .`

## Running mypy
`poetry add --dev mypy`  
`poetry run mypy .`

### Install
```
uv venv
uv sync
source .venv/bin/activate
uv pip install -e .
```
### Run 
1. Run data generation - generate data for tests - you can do it once (before models tests)

Syntax:
```./gen_data.sh <model_name> <model_path> <test_name>```

Example:
```./gen_data.sh Bielik-11B-v3 /net/models synthetic```

3. Run evals
```./run_eval Bielik-11B-v3 /net/models synthetic```

### Config
1. config_models.sh - model config - context length, temperature, top_k, top_p and chat template (now is base model; change to bielik-caht if instruct model)
2. config_tasks.sh - task config
3. synthetic.yaml - test config (set of tasks) 


## Survey agent
## set up openchat model https://github.com/imoneoi/openchat
```
pip3 install ochat
python -m ochat.serving.openai_api_server --model openchat/openchat_3.5  --engine-use-ray --worker-use-ray --tensor-parallel-size N
```
You can choose the nums of gpu using parameter  "--tensor-parallel-size "

# Survey agent
## set up openchat model https://github.com/imoneoi/openchat
```
pip3 install ochat
python -m ochat.serving.openai_api_server --model openchat/openchat_3.5  --engine-use-ray --worker-use-ray --tensor-parallel-size 1
```
You can choose the num of gpus using parameter  "--tensor-parallel-size "
## Tokenizer
Need model tokenizer for calculating tokens of chunk or controlling


## set up arxiv-sanity

先运行 feature_func.py 生成一下 tfidf 要用的 features.p，之后运行 arxiv_sanity_func.py 就可以

## set up config file


## Use GPT 4
You are supposed to export OPENAI_API_KEY="your_key" before use GPT-4 model

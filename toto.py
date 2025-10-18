from abstractcore import create_llm
llm = create_llm('lmstudio', model='qwen/qwen3-next-80b')
response = llm.generate('What is in this image?', media=['photo.jpg'])
print(response)

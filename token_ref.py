from cliponnx.simple_tokenizer import SimpleTokenizer
samples=['a photo of a cat','Hello, world!','An astronaut riding a horse']
st=SimpleTokenizer()
print('Python reference:')
for s in samples:
    ids=st.encode(s)
    print(repr(s),'=>',ids,'len=',len(ids))
print('Special',st.encoder['<|startoftext|>'],st.encoder['<|endoftext|>'])

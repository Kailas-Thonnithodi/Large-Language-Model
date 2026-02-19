import tiktoken

'''
Has basically a similar structure to SimpleTokeniserV2, however implements the BPE algorithm for deal with unknown values and certain string wrappers and separators.
'''

if __name__ == "__main__":
    tiktokenisor = tiktoken.get_encoding("o200k_base")
    integers = tiktokenisor.encode("Hello World world")
    print(sum(integers))
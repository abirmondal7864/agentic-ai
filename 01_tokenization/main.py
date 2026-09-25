import tiktoken

enc =tiktoken.encoding_for_model("gpt-3.5-turbo")
text="Hey there! I am Abir Mondal"
tokens=enc.encode(text)
# Tokens :  [19182, 1070, 0, 358, 1097, 3765, 404, 51972, 278]
print("Tokens : ", tokens)

decoded =enc.decode([19182, 1070, 0, 358, 1097, 3765, 404, 51972, 278])
print("Decoded : ", decoded)

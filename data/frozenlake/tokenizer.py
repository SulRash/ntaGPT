# Map-to-Token to avoid the tokenization step.
# 0, 1, and 2 are reserved for <eor>, <eom>, and \n respectively
MTT = {
    "S": "3",
    "G": "4",
    "F": "5",
    "H": "6"
}

TTM = {
    0: "<eor>",
    1: "<eom>",
    2: "\n",
    3: "S",
    4: "G",
    5: "F",
    6: "H"
}

def tokenizer(string: str):
    string = string.replace("<eor>", "0")
    string = string.replace("<eom>", "1")
    string = string.replace("\n", "2")
    string = string.replace("S", MTT["S"])
    string = string.replace("G", MTT["G"])
    string = string.replace("F", MTT["F"])
    string = string.replace("H", MTT["H"])
    return list(map(int, list(string)))

def detokenizer(ids):
    ids = ids.tolist()[0]
    output = ''.join([TTM[i] for i in ids])
    return output
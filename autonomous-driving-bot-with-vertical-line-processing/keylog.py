import win32api

keyList = ["\b"]
for char in "WASD":
    keyList.append(char)

def getkeys():
    keys = []
    for key in keyList:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

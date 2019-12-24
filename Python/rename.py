from os import listdir
from os import rename

#filePath = 'D:\\Workspace\\core-10.0\\workspace\\'
#module = 'kanban'
#localPath = '\\server\\translation'

filePath = '\\\\dhselk2\\translation-new'

#for f in listdir(filePath + module + localPath):
for f in listdir(filePath):
    fold = f
    fnew = f.lower()
    print (fold)
    rename(filePath + '\\' + fold, filePath + '\\' + fnew)


import os
import re

INC_PATH="D:\Tmp\ST\spl-splitter-master\spl-splitter-master\STM8S_StdPeriph_Driver\inc"
SRC_PATH="D:\Tmp\ST\spl-splitter-master\spl-splitter-master\STM8S_StdPeriph_Driver\src"
OUT_PATH="D:\Tmp\ST\spl-splitter-master\spl-splitter-master\STM8S_StdPeriph_Driver\src_\\"

base_string = "#include {0} \n\r {1}"

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def Process_C_file(content, header_file):
    count = 0
    prev_position = -1
    #print(content)
    #match_object = re.finditer(r'(\/\*\*)', content)
    for match in re.finditer(r'(\/\*\*)', content):
        #print(match.span())
        if prev_position == -1:
            prev_position = match.span()[0]
        else:                        
            count = count + 1
            #print(prev_position)
            #print(match.span()[1])
            file_content = content[prev_position: (match.span()[1]-3)]
            file_content = base_string.format(header_file, file_content)
            prev_position = match.span()[0]
            source_file_name = header_file.removesuffix(".h") + "_{}_.c".format(count)
            #print(source_file_name)
            new_source = open(OUT_PATH + source_file_name, "w")
            new_source.write(file_content)
            new_source.close()

count = 0        
for header_file in os.listdir(INC_PATH):
    if header_file.endswith(".h"):        
        print(header_file.removesuffix(".h"))
        found = find(header_file.removesuffix(".h")+".c", SRC_PATH)
        if found is not None:
            print(found)
            source_file = open(found, "r")
            Process_C_file(source_file.read(), header_file)
            
            print("Done "+header_file)

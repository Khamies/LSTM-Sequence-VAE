import csv  
def get_indent(string):
    if (len(string) == 0):
        return 0 
    indent = 0 
    char = string[0]
    i = 1 
    while(char.isspace()):
        if(char == "\t"):
            indent += 4 
        else:
            indent += 1 
        char = string[i]
        i += 1 
    return indent

def get_sprite_types(sprites): 
    sprite_types = {}
    last_type = "None"
    last_index = 0 
    for sprite in sprites:
        parts = sprite.split(">")
        name = parts[0].strip()
        if len(name) == 0:
            pass 
        else:
            indent = get_indent(parts[0])
            
            if len(parts) < 2:
                if (indent >= last_index):
                    t = last_type 
                else:
                    t = "parent"
        
            else:
                params = parts[1].split(" ")
                t = params[0]
                i = 1
                while len(t) == 0 and i < len(params):
                    t = params[i]
                    i+=1 

                if len(t) == 0:
                    t = "parent"

                elif  t[0].islower():
                    if (indent >= last_index):
                        t = last_type 
                    else:
                        t = "parent"
            
             
            if t in sprite_types:
                sprite_types[t].append(name)
            else: 
                sprite_types[t] = [name]
            
            last_type = t
            last_index = indent

    
    return sprite_types


def get_general_names(sprites):
    types = get_sprite_types(sprites) 
    name_dict = {}

    for t, names in types.items():
        for i, name in enumerate(names):
            name_dict[name] = t.lower() + str(i)
    return name_dict   

def _replace_parts(name, new_name):
    parts = []
    parts.append((" " + name + " ", " " + new_name + " "))
    parts.append((" " + name + "\n", " " + new_name + "\n"))
    parts.append(("=" + name + " ", "=" + new_name + " "))
    parts.append(("=" + name + "\n", "=" + new_name + "\n"))
    parts.append(("\t" + name + " ", "\t" + new_name + " "))
    parts.append(("\t" + name + "\n", "\t" + new_name + "\n"))
    return parts 


def change_names(description, name_dict):
    items = list(name_dict.keys())
    items.sort(key = lambda x: len(x), reverse=True)
    for key in items:
        for old, new in _replace_parts(key, name_dict[key]):
            description = description.replace(old, new)
    return description

def get_sprites(description):
    in_sprites = False 
    sprites = []
    for line in description.split("\n"): 
        if not in_sprites:
            if("SpriteSet" in line):
                in_sprites = True 

        else:
            if ("LevelMapping" in line or "InteractionSet" in line or "TerminationSet" in line):
                in_sprites = False 
            else:
                sprites.append(line)
    return sprites 

def generalize_sprites(desc):
  
    sprites = get_sprites(desc)
    new_names = get_general_names(sprites)
    return change_names(desc, new_names)


def generalize_examples(csv_file):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            desc  = open("VGDLDataGeneralized/" + row[1], "r").read() 
            new_file = open("VGDLDataGeneralized/" + row[1], "w") 
            new_file.write(generalize_sprites(desc))
   
if __name__ == "__main__":
    generalize_examples("VGDLDataGeneralized/examples/all_games_sp.csv")
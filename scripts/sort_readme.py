input_filename = 'README.md'
output_filename = 'SORTED_README.md'

language_mark = '##'
theme_mark = '####'
start_mark = '<!-- /MarkdownTOC -->'
item_mark = '* ['

line = ''
items_list = []

with open(input_filename) as f_in, open(output_filename, 'w') as f_out:
    
    for line in f_in:
        
        if line.startswith(item_mark):
            items_list.append(line)
            
        else:
            for item in sorted(items_list, key=lambda x: x.lower()):
                f_out.write(item)
            items_list = []
            f_out.write(line)
    for item in sorted(items_list, key=lambda x: x.lower()):
                f_out.write(item)
             


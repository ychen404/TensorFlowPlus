replacements = {'BATCH_SIZE = 1':'BATCH_SIZE = 4', 'graph_conv_16_': 'graph_conv_16_bat_4_'}
with open('conv_16_64.py') as infile, open('conv_16_64_bat_4.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 1':'BATCH_SIZE = 4', 'graph_conv_12_': 'graph_conv_12_bat_4_'}
with open('conv_12_64.py') as infile, open('conv_12_64_bat_4.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 1':'BATCH_SIZE = 4', 'graph_conv_8_': 'graph_conv_8_bat_4_'}
with open('conv_8_64.py') as infile, open('conv_8_64_bat_4.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 1':'BATCH_SIZE = 4', 'graph_conv_4_': 'graph_conv_4_bat_4_'}
with open('conv_4_64.py') as infile, open('conv_4_64_bat_4.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 1':'BATCH_SIZE = 4', 'graph_conv_2_': 'graph_conv_2_bat_4_'}
with open('conv_2_64.py') as infile, open('conv_2_64_bat_4.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

#replacements = {'BATCH_SIZE = 1':'BATCH_SIZE = 4', 'graph_conv_1_': 'graph_conv_1_bat_4_'}
#with open('conv_1_64.py') as infile, open('conv_1_64_bat_4.py', 'w') as outfile:
#    for line in infile:
#        for src, target in replacements.iteritems():
#            line = line.replace(src, target)
#        outfile.write(line)

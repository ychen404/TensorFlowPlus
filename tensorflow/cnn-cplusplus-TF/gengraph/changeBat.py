replacements = {'BATCH_SIZE = 4':'BATCH_SIZE = 8', 'graph_fc_16_bat_4_': 'graph_fc_16_bat_8_'}
with open('fc_16_64_bat_4.py') as infile, open('fc_16_64_bat_8.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 4':'BATCH_SIZE = 8', 'graph_fc_12_bat_4_': 'graph_fc_12_bat_8_'}
with open('fc_12_64_bat_4.py') as infile, open('fc_12_64_bat_8.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 4':'BATCH_SIZE = 8', 'graph_fc_8_bat_4_': 'graph_fc_8_bat_8_'}
with open('fc_8_64_bat_4.py') as infile, open('fc_8_64_bat_8.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 4':'BATCH_SIZE = 8', 'graph_fc_4_bat_4_': 'graph_fc_4_bat_8_'}
with open('fc_4_64_bat_4.py') as infile, open('fc_4_64_bat_8.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 4':'BATCH_SIZE = 8', 'graph_fc_2_bat_4_': 'graph_fc_2_bat_8_'}
with open('fc_2_64_bat_4.py') as infile, open('fc_2_64_bat_8.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

replacements = {'BATCH_SIZE = 4':'BATCH_SIZE = 8', 'graph_fc_1_bat_4_': 'graph_fc_1_bat_8_'}
with open('fc_1_64_bat_4.py') as infile, open('fc_1_64_bat_8.py', 'w') as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)

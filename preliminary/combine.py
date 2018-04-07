import json
from collections import defaultdict

files = ["resultsQTFT", "resultsPGATFT2", "resultsPGAPGA", "resultsQQ"]
titles = ["Q-agent vs Q-agent", "PGA-PP vs PGA-PP"]

# "Q-agent vs Q-agent", "PGA-PP vs PGA-PP"
cols = ["CC", "AC", "CD", "AA", "AD", "DD"]

all = {}
for f in files:
    with open(f + '.json') as json_data:
        d = json.load(json_data)
        for k, v in d.items():
            if k not in all:
                all[k] = {}

            for k2, v2 in v.items():
                all[k][k2] = v2

header = ''.join(["," + key + ",,,,," for key in titles])
subheader = ''.join(["," +k for k in cols*len(titles)])
my_str = header + "\n" + subheader
print(my_str)
for g in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9995]:
    row = str(g)
    for t in titles:
        for c in cols:
            if c not in all[str(g)][t]:
                row += ",-"
            else:
                row += "," + str(all[str(g)][t][c])
    print(row)

#
# for all[0][k].items()
#
# merged_all = {key: value for (key, value) in ( + all[1][k].items() + all[2][k].items() + all[3][k].items()) for k in all[0].keys()}
#
# header = ["," + key + "," for (key, value) in merged_all[0.1].items()]
# print(header)
#
#
# # for g in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9995]:
# #     merged_all[g]
# #     results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
# #     all[0][g].values()[0], all[1][g].values()[0], all[2][g].values()[0], all[3][g].values()[0]




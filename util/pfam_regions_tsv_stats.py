#! /bin/usr/python3
# Copyright (C) 2017  Han Lin <hotdogee [at] gmail [dot] com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
import os
import sys
from collections import defaultdict

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar_str = fill * filled_length + '　' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar_str, percent, suffix), end='\r', file=sys.stderr)
    # Print New Line on Complete
    if iteration == total:
        print(file=sys.stderr)

def pfam_regions_tsv_stats(tsv_file, bin_size=10):
    """Prints Pfam-A.regions.uniprot.tsv file stats
    tsv_file: string path to tsv file.
    bin_size: bin size for region length histogram
    """
    tsv_f = open(tsv_file, 'r')
    file_size = os.path.getsize(tsv_file)

    regions_per_seq = defaultdict(int)
    seqs_per_domain = defaultdict(int)
    len_list = []
    hist_list = []

    bytes_read = 0
    print_progress_bar(0, file_size, prefix='Progress:', suffix='Complete', length=50)
    line_num = 0
    for line in tsv_f:
        bytes_read += len(line)
        line_num += 1
        if line_num == 1: continue # skip header
        tokens = line.strip().split()
        seq_id = '{}.{}'.format(tokens[0], tokens[1])
        regions_per_seq[seq_id] += 1
        seqs_per_domain[tokens[4]] += 1
        region_len = int(tokens[6]) - int(tokens[5])
        region_bin = int(region_len / bin_size)
        len_list.append(region_len)
        # extend hist_list if needed
        if region_bin + 1 > len(hist_list):
            hist_list += [0] * (region_bin + 1 - len(hist_list))
        hist_list[region_bin] += 1
        if len(len_list) % 100000 == 0:
            print_progress_bar(bytes_read, file_size, prefix='Progress:', suffix='Complete', length=50)
    print_progress_bar(file_size, file_size, prefix='Progress:', suffix='Complete', length=50)
    tsv_f.close()

    # output
    print('{}\t{}'.format('Bin', 'Count'))
    for i, c in enumerate(hist_list):
        print('{}\t{}'.format(i * bin_size, c))
    print('{:>20}{:>15}'.format('Region Count:', len(len_list)))
    print('{:>20}{:>15}'.format('Sequence Count:', len(regions_per_seq)))
    print('{:>20}{:>15}'.format('Domain Count:', len(seqs_per_domain)))
    print('Region length:')
    len_list = sorted(len_list)
    print('{:>20}{:>15}'.format('Min:', len_list[0]))
    print('{:>20}{:>15}'.format('Median:', len_list[int(len(len_list)/2)]))
    print('{:>20}{:>15}'.format('Average:', int(sum(len_list)/len(len_list))))
    print('{:>20}{:>15}'.format('Max:', len_list[-1]))
    print('Regions per sequence:')
    len_list = sorted(regions_per_seq.values())
    print('{:>20}{:>15}'.format('Min:', len_list[0]))
    print('{:>20}{:>15}'.format('Median:', len_list[int(len(len_list)/2)]))
    print('{:>20}{:>15}'.format('Average:', int(sum(len_list)/len(len_list))))
    print('{:>20}{:>15}'.format('Max:', len_list[-1]))
    print('Sequences per domain:')
    len_list = sorted(seqs_per_domain.values())
    print('{:>20}{:>15}'.format('Min:', len_list[0]))
    print('{:>20}{:>15}'.format('Median:', len_list[int(len(len_list)/2)]))
    print('{:>20}{:>15}'.format('Average:', int(sum(len_list)/len(len_list))))
    print('{:>20}{:>15}'.format('Max:', len_list[-1]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prints Pfam-A.regions.uniprot.tsv file stats.')
    parser.add_argument('-b', '--binsize', type=int, default=10,
                        help='Bin size for region length histogram')
    parser.add_argument('tsv_file', type=str,
                        help='Path to the tsv file')

    args = parser.parse_args()
    pfam_regions_tsv_stats(args.tsv_file, bin_size=args.binsize)

# Pfam-A.regions.uniprot.tsv.gz
#        Region Count:       88761542
#      Sequence Count:       54223493
#        Domain Count:          16712
# Region length:
#                 Min:              3
#              Median:            127
#             Average:            156
#                 Max:           2161
# Regions per sequence:
#                 Min:              1
#              Median:              1
#             Average:              1
#                 Max:            572
# Sequences per domain:
#                 Min:              2
#              Median:            863
#             Average:           5311
#                 Max:        1078482
# Bin     Count
# 0       10736
# 10      507548
# 20      1862256
# 30      2839796
# 40      3513480
# 50      4427920
# 60      5133762
# 70      4863518
# 80      4439422
# 90      4892708
# 100     4314578
# 110     4519519
# 120     3954749
# 130     3526058
# 140     3465305
# 150     2950855
# 160     2858057
# 170     2964424
# 180     2684697
# 190     2318746
# 200     2153935
# 210     1942455
# 220     1449940
# 230     1551969
# 240     1336736
# 250     1239419
# 260     1050417
# 270     1017016
# 280     968286
# 290     931334
# 300     797207
# 310     728925
# 320     672457
# 330     598417
# 340     612010
# 350     483958
# 360     524319
# 370     434130
# 380     371327
# 390     367366
# 400     316456
# 410     291056
# 420     265427
# 430     252710
# 440     298811
# 450     245771
# 460     226116
# 470     135676
# 480     134706
# 490     141289
# 500     98976
# 510     96327
# 520     70990
# 530     91343
# 540     96918
# 550     60021
# 560     44178
# 570     33404
# 580     32258
# 590     29928
# 600     29027
# 610     24304
# 620     21672
# 630     16949
# 640     26890
# 650     20743
# 660     16642
# 670     14189
# 680     15196
# 690     52424
# 700     11438
# 710     15999
# 720     8725
# 730     10485
# 740     45731
# 750     44675
# 760     4827
# 770     4548
# 780     4757
# 790     3872
# 800     5665
# 810     4295
# 820     4133
# 830     3423
# 840     2453
# 850     2009
# 860     1751
# 870     2141
# 880     1886
# 890     1704
# 900     1049
# 910     1194
# 920     972
# 930     887
# 940     944
# 950     749
# 960     799
# 970     702
# 980     1354
# 990     3378
# 1000    11399
# 1010    11037
# 1020    15639
# 1030    9046
# 1040    3352
# 1050    1574
# 1060    767
# 1070    1941
# 1080    2880
# 1090    864
# 1100    1566
# 1110    1097
# 1120    1154
# 1130    1181
# 1140    1353
# 1150    1252
# 1160    2825
# 1170    2895
# 1180    1016
# 1190    847
# 1200    592
# 1210    417
# 1220    321
# 1230    281
# 1240    430
# 1250    282
# 1260    179
# 1270    147
# 1280    344
# 1290    521
# 1300    321
# 1310    171
# 1320    107
# 1330    80
# 1340    85
# 1350    249
# 1360    198
# 1370    92
# 1380    56
# 1390    63
# 1400    52
# 1410    95
# 1420    41
# 1430    56
# 1440    26
# 1450    54
# 1460    147
# 1470    431
# 1480    580
# 1490    236
# 1500    254
# 1510    332
# 1520    760
# 1530    857
# 1540    273
# 1550    321
# 1560    243
# 1570    61
# 1580    40
# 1590    64
# 1600    16
# 1610    79
# 1620    56
# 1630    109
# 1640    145
# 1650    77
# 1660    38
# 1670    93
# 1680    10
# 1690    4
# 1700    6
# 1710    16
# 1720    9
# 1730    19
# 1740    22
# 1750    10
# 1760    13
# 1770    42
# 1780    24
# 1790    14
# 1800    15
# 1810    11
# 1820    4
# 1830    9
# 1840    2
# 1850    2
# 1860    7
# 1870    4
# 1880    3
# 1890    8
# 1900    3
# 1910    3
# 1920    11
# 1930    28
# 1940    31
# 1950    20
# 1960    6
# 1970    5
# 1980    2
# 1990    1
# 2000    0
# 2010    18
# 2020    62
# 2030    205
# 2040    19
# 2050    16
# 2060    1
# 2070    0
# 2080    0
# 2090    0
# 2100    0
# 2110    0
# 2120    0
# 2130    0
# 2140    0
# 2150    0
# 2160    1
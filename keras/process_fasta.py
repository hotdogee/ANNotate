#! /bin/usr/python
import csv
from collections import defaultdict


def pfam_fa_to_domain_dict(fa_path):
    domain_dict = defaultdict(list)
    domain, sequence = '', ''
    with open(fa_path, 'r') as fa_f:
        for line in fa_f:
            line = line.strip()
            if len(line) > 0 and line[0] == '>':
                if sequence:
                    domain_dict[domain].append(sequence)
                    domain_id, domain_name, sequence = '', '', ''
                # parse header
                domain = line.split()[2]
            else:
                sequence += line
    return domain_dict

# max sequence length = 2162
# median sequence length = 129
def sequence_list(domain_dict):
    sequence_list = []
    i = 0
    for dom in sorted(domain_dict, key=lambda k: (len(domain_dict[k]), k), reverse=True):
        for seq in domain_dict[dom]:
            sequence_list.append(seq)
    return sequence_list


def domain_dict_count(domain_dict, count=None):
    domain_counts = []
    i = 0
    for k in sorted(domain_dict, key=lambda k: (len(domain_dict[k]), k), reverse=True):
        domain_counts.append([k, len(domain_dict[k])])
        i += 1
        if count and i >= count:
            break
    return domain_counts


def domain_dict_csv(domain_dict, csv_path):
    with open(csv_path, 'w') as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=['domain', 'sequence'])
        writer.writeheader()
        for k in sorted(domain_dict, key=lambda k: (len(domain_dict[k]), k), reverse=True):
            for sequence in domain_dict[k]:
                writer.writerow({'domain': k, 'sequence': sequence})


if __name__ == "__main__":
    fa_path = 'datasets/Pfam-A.fasta'
    csv_path = 'datasets/Pfam-A.fasta.csv'

    domain_dict = pfam_fa_to_domain_dict('datasets/Pfam-A.fasta')
    print('Number of domains: ', len(domain_dict))
    # Number of domains:  16479
    domain_counts = domain_dict_count(domain_dict)
    print('Number of domains with more than 1,000 sequences:',
          len(list(filter(lambda x: x[1] > 1000, domain_counts))))
    # Number of domains with more than 1,000 sequences: 3605
    print('Number of domains with more than 10,000 sequences:',
          len(list(filter(lambda x: x[1] > 10000, domain_counts))))
    # Number of domains with more than 10,000 sequences: 378

    # write csv
    domain_dict_csv(domain_dict, csv_path)

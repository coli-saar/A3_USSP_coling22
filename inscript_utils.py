"""
misc utils for InScript
"""

from collections import namedtuple

irregular_prefixes = \
    {'event': ['unrelev_', 'relnscrev_', 'screv_other', 'unclear_', 'irregular', 'evoking'],
     # note: evoking is excluded here because it has irregular head annotations which does not fulfil our hypothesis
     'participant': ['npart_', 'no_label', 'scrpart_other', 'suppvcomp', 'unclear',
                     'head_of_partitive', '@partitive/quantitative']}

data_fields = ['content', 'id', 'pos', 'dep_head', 'participant',
               'part_head', 'event', 'event_head', 'coreference_chain_idx']


protagonists = {'ScrPart_bather_bath', 'ScrPart_rider_bicycle', 'ScrPart_passenger_bus', 'ScrPart_cook_cake',
                'ScrPart_passenger_flight', 'ScrPart_shopper_grocery', 'ScrPart_customer_haircut',
                'ScrPart_customer_library', 'ScrPart_passenger_train', 'ScrPart_gardener_tree', 'protagonist'}


def is_regular(label: str, modes=('event', 'participant')):
    label = label.lower()
    for mode in modes:
        for null_label in irregular_prefixes[mode]:
            if null_label in label:
                return False
    return True


def is_protagonist(partpcipant: str):
    for prot in protagonists:
        if prot.lower() in partpcipant.lower():
            return True
    return False


def line_named_tuple(mode: str):
    fields = [data_fields[i] for i in range(len(data_fields)) if 1 == 0 or mode[i] == 1]
    return namedtuple('line', fields)



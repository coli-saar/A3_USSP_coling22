"""
dataset reader for InScript as a sequence labeling task
sampler to make sure instances in each batch are from the same scenario
"""

import os
from typing import List, Iterable
import itertools

from allennlp.data import Instance, AllennlpDataset
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from global_constants import CONST
from misc import QualityOfPythonLife


class InScriptSequenceLabelingReader(DatasetReader):
    """
    reads InScript data as a sequence labeling task of event and/or participant identification.
    Sequence labeling could either be performed for the entire story, or only for events/participant tokens.
    expects data output formated ?00011111 (see the template paramter in InScript.py).

    note: this reader merges all irregular event categories as scenario-aware 'irregular's.

    parameters:
        candidate_types: a non-empty subset of {'events', 'participants'}, indicating which instances will be loaded.
        mode:
            'normal': load all regular and irregular labels
            'regular_only': load only regular labels
            'regular_identification': load both regular and irregular events, but merge all regular events

    fields:
        story: text field, the story.
        scr_labels: TextField. the label of each token in the story field.
        squeezed_labels: TextField. a sublist of scr_labels that corresponds to dependency heads of VPs and NPs, which
        are to receive a label.
        label_indices: MetadataField. A list that collects the indices of the squeezed_labels in scr_labels.
        scenario: TextField. the current scenario.
    """
    def __init__(self, candidate_types, word_indexer, mode='normal'):
        super().__init__()
        self.candidate_types = candidate_types
        self.mode = mode
        self.word_indexer = word_indexer
        self.label_indexer = {'scr_labels': SingleIdTokenIndexer(namespace='scr_labels')}
        self.scenario_indexer = {'scenarios': SingleIdTokenIndexer(namespace='scenarios')}
        self.data = {}

    def _is_regular(self, label: str):
        label = label.lower()
        for candidate_type in self.candidate_types:
            for null_label in CONST.irregular_prefixes[candidate_type]:
                if null_label in label:
                    return False
        return True

    def text_to_instance(self,
                         story: List[str],
                         label_indices: List[int],
                         scenario: str,
                         scr_labels: List[str] = None) -> Instance:
        story_field = TextField([Token(t) for t in story], token_indexers=self.word_indexer)
        scr_label_indices_field = MetadataField(label_indices)
        scenario_field = TextField([Token(scenario)], token_indexers=self.scenario_indexer)
        if scr_labels is not None:
            labels_field = TextField([Token(t) for t in scr_labels], token_indexers=self.label_indexer)
            squeezed_labels_field = TextField([Token(t) for t in scr_labels if t != CONST.null_label],
                                              token_indexers=self.label_indexer)
            return Instance({'story': story_field,
                             'scr_labels': labels_field,
                             'squeezed_labels': squeezed_labels_field,
                             'label_indices': scr_label_indices_field,
                             'scenario': scenario_field})
        else:
            return Instance({'story': story_field,
                             'label_indices': scr_label_indices_field,
                             'scenario': scenario_field})

    def _read(self, folder: str) -> Iterable[Instance]:
        """
        Note: for transformer-tokenized data, dummy labels with 'X' will be be replaced by 'none' thus will NOT be
         added to valid labels. So no further processing is needed for the evaluation of loss.
        :param folder:
        :return:
        """
        for scenario in os.listdir(folder):
            # sometimes we need to deep copy the buffers when yielding them
            story_buffer = ['{}_{}'.format(CONST.begin_of_story_type, scenario)]
            labels_buffer = [CONST.null_label]
            scenario_name = scenario if '_' not in scenario else scenario[:scenario.index('_')]

            for line in open(os.path.join(folder, scenario), 'r'):
                splited_line = line.split()
                if len(splited_line) < 2 and CONST.end_of_story_type not in line:
                    continue
                story_buffer.append(splited_line[0])
                if line.__contains__(CONST.end_of_story_type):
                    labels_buffer.append(CONST.null_label)
                    # end of story line, yield instance and clear buffers.
                    label_indices = [i for i in range(len(labels_buffer)) if labels_buffer[i] != CONST.null_label]
                    yield self.text_to_instance(
                        story=story_buffer,
                        scr_labels=labels_buffer,
                        label_indices=label_indices,
                        scenario=scenario)  # if '_' not in scenario else scenario[:scenario.index('_')])
                    story_buffer = ['{}_{}'.format(CONST.begin_of_story_type, scenario)]
                    labels_buffer = [CONST.null_label]
                    continue
                elif 'participants' in self.candidate_types and splited_line[2] in ['single', 'yes']:
                    # participant head
                    def _normal_participant_label():
                        if CONST.merge_irregular_labels:
                            if not self._is_regular(splited_line[1]):
                                # irregular participant
                                return f'{CONST.irregular_participant_label}_{scenario_name}'
                            else:
                                return splited_line[1]
                        else:
                            return splited_line[1]

                    if self.mode == 'normal':
                        labels_buffer.append(_normal_participant_label())
                    elif self.mode in ['regular_only', 'regular_only_for_ussp']:
                        if self._is_regular(splited_line[1]):
                            labels_buffer.append(_normal_participant_label())
                        else:
                            labels_buffer.append(CONST.null_label)
                    elif self.mode == 'regular_identification':
                        if self._is_regular(splited_line[1]):
                            labels_buffer.append(f'{CONST.regular_participant_label}')
                        else:
                            labels_buffer.append(f'{CONST.irregular_participant_label}')
                    else:
                        raise Exception('Invalid loading mode. Check the data loader doc.')

                elif 'events' in self.candidate_types and splited_line[4] in ['single', 'yes']:
                    # event head
                    def _normal_event_label():
                        if CONST.merge_irregular_labels:
                            if not self._is_regular(splited_line[3]):
                                # irregular participant
                                return f'{CONST.irregular_event_label}_{scenario_name}'
                            else:
                                return splited_line[3]
                        else:
                            return splited_line[3]

                    if self.mode == 'normal':
                        labels_buffer.append(_normal_event_label())
                    elif 'regular_only' in self.mode:
                        if self._is_regular(splited_line[3]):
                            labels_buffer.append(_normal_event_label())
                        else:
                            labels_buffer.append(CONST.null_label)
                    elif self.mode == 'regular_identification':
                        if self._is_regular(splited_line[3]):
                            labels_buffer.append(f'{CONST.regular_event_label}')
                        else:
                            labels_buffer.append(f'{CONST.irregular_event_label}')
                    else:
                        raise Exception('Invalid loading mode. Check the data loader doc.')

                else:
                    labels_buffer.append(CONST.null_label)

    def split(self, validation_scenario, test_scenario, data_instances):
        if len(self.data) == 0:
            for scenario in CONST.scenario_s:
                self.data[scenario] = \
                    [inst for inst in data_instances if inst.fields['scenario'].tokens[0].text == scenario]

        """ split data for the sake of cross-validation """
        train_scenarios = [scenario for scenario in CONST.scenario_s
                           if scenario not in [test_scenario, validation_scenario]]
        train_data = AllennlpDataset(list(itertools.chain(*[self.data[scenario] for scenario in train_scenarios])))
        validation_data = AllennlpDataset(self.data[validation_scenario])
        test_data = AllennlpDataset(self.data[test_scenario])
        return train_data, validation_data, test_data

    @staticmethod
    def scenario_of_label(label: str):
        """
        return the scenario of a specific label
        """
        if '_' not in label:
            return '@UNK'
        else:
            return label[QualityOfPythonLife.last_index_of(label, '_') + 1:]


def data_split(path=os.path.join('.', 'clean_data_xlnet_000011111'),
               train_path=os.path.join('.', 'data_train'),
               val_path=os.path.join('.', 'data_val'),
               test_path=os.path.join('.', 'data_test'),
               ratio: List[int] = None):
    """
    split the data into train and val. Test will be performed with cross validation.
    use only on outputs from InScript.py.
    :return:
    """
    if ratio is None:
        ratio = [8, 1, 1]

    files = os.listdir(path)
    for folder in [train_path, val_path, test_path]:
        if not os.path.exists(folder):
            os.mkdir(folder)
    for file in files:
        filepath = os.path.join(path, file)
        trainfilepath, valfilepath, testfilepath = \
            os.path.join(train_path, file), os.path.join(val_path, file), os.path.join(test_path, file)
        with open(filepath, 'r') as fin:
            train_out, val_out, test_out = \
                open(trainfilepath, 'w'), open(valfilepath, 'w'), open(testfilepath, 'w')
            writer = train_out
            counter = 0
            inline = fin.readline()
            while inline != '':
                if '<end_of_story>' in inline:
                    writer.write(inline)
                    counter += 1
                    if counter % sum(ratio) < ratio[0]:
                        writer = train_out
                    elif counter % sum(ratio) < ratio[0] + ratio[1]:
                        writer = val_out
                    else:
                        writer = test_out
                    inline = fin.readline()
                else:
                    writer.write(inline)
                    inline = fin.readline()
            train_out.close(), val_out.close(), test_out.close()


if __name__ == '__main__':
    data_split()

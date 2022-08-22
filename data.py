"""
dataset readers for InScript as a clustering task
sampler to make sure instances in each batch are from the same scenario
"""

import os
from typing import List, Iterable
from collections import namedtuple
import random
import math
import itertools
import dill
import spacy

from allennlp.data import Instance
from allennlp.data.fields import TextField, MetadataField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset
from allennlp.data.samplers.samplers import BatchSampler

from global_constants import CONST
from misc import QualityOfPythonLife
import inscript_utils


class SequenceLabellingReader(DatasetReader):
    def __init__(self, word_indexer, label_type):
        super().__init__()
        self.word_indexer = word_indexer
        self.label_type = label_type

    def text_to_instance(self, text: List[str], labels: List[str]) -> Instance:
        text_field = TextField([Token(t) for t in text], token_indexers=self.word_indexer)
        labels_field = TextField([Token(lb) for lb in labels],
                                 token_indexers={'seq_labels': SingleIdTokenIndexer(namespace='seq_labels')})
        return Instance({'story': text_field,
                         'seq_labels': labels_field})

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as fin:
            lines = list(fin)
            for i in range(0, len(lines), 4):
                text = lines[i].split()
                if self.label_type == 'pos':
                    labels = lines[i + 1].split()
                elif self.label_type == 'ner':
                    labels = lines[i + 2].split()
                else:
                    labels = lines[i + 3].split()
                assert len(text) == len(labels)
                yield self.text_to_instance(text, labels)


class TextClassificationReader(DatasetReader):
    def __init__(self, word_indexer):
        super().__init__()
        self.word_indexer = word_indexer

    def text_to_instance(self, story: List[str], label: int) -> Instance:
        story_field = TextField([Token(t) for t in story], token_indexers=self.word_indexer)
        label_field = LabelField(label, skip_indexing=True)
        return Instance({'story': story_field, 'label': label_field})

    def _read(self, file_path: str):
        with open(file_path, 'r') as fin:
            for line in fin:
                story_str, label_str = line.split('\t')
                story = story_str.split()
                label = int(label_str)
                yield self.text_to_instance(story, label)


class StoryInstanceReader(DatasetReader):
    """
    read InScript from InScript format ?00001111.
    each instance corresponds to a whole story.
    when a token receives both event and participant annotations, only the event annotation is kept
    fields:
        story: text field
        event_indices: metadata field List[Int], index of event candidates
        event_labels: event annotations, matching candidates specified in 'event_indicies'
        participant_indices: metadata field List[Int], index of participant candidates
        participant_labels: participant annotations, matching candidates specified in 'participant_indicies'
        scenario: text field, with 1 token
        story_id: metadata field Int, universal id of the story
        coref_chain_idx: metadata field List[str], index of the coref chain, defaults to 'none'
        dep_marks: metadata field List[List[int]], indices of the candidate clusters that the current token has dependency with
    """
    def __init__(self, word_indexer,
                 include_scenario_phrase,
                 include_irregular=False,
                 regularity_prediction_folder=None,
                 ):
        """
        """
        super().__init__()
        ''' the intended external word indexer is one corresponds to a pretrained transformer model '''
        self.word_indexer = word_indexer
        self.include_irregular = include_irregular
        ''' we hold the possibility to perform representation learning for scenarios '''
        self.scenario_indexer = {'scenarios': SingleIdTokenIndexer(namespace='scenarios')}
        self.cluster_name_indexer = {'scr_labels': SingleIdTokenIndexer(namespace='scr_labels')}
        # self.coref_indexer = {'coref_idx': SingleIdTokenIndexer(namespace='coref_idx')}
        self.scenarios = list()
        self.regularity_prediction_folder = regularity_prediction_folder
        self.include_scenario_phrase = include_scenario_phrase
        self.data = dict()

    @staticmethod
    def context(_instance, index, width=12):
        list_of_tokens = _instance.fields['story'].tokens
        list_of_txt_tokens = [t.text.replace('▁', '') for t in list_of_tokens]
        return StoryInstanceReader.context_from_list(text=list_of_txt_tokens, index=index, width=width)

    @staticmethod
    def context_from_list(text: list, index: int, width=12):
        clean_text = [t.replace('_', '') for t in text]
        start_index = max(index - width, 0)
        end_index = min(index + width, len(clean_text) - 1)
        tokens = [clean_text[i] if i != index else f'[{clean_text[i]}]' for i in range(start_index, end_index)]

        return ' '.join(tokens)

    def text_to_instance(self,
                         story: List[str],
                         event_indices: List[int],
                         event_labels: List[str],
                         participant_indices: List[int],
                         participant_labels: List[str],
                         merged_indices: List[int],
                         merged_labels: List[str],
                         scenario: str,
                         story_id: int,
                         coreference_chain_idx: List[str],
                         dep_marks: List[str]) -> Instance:
        story_field = TextField([Token(t) for t in story], token_indexers=self.word_indexer)
        event_indices_field = MetadataField(event_indices)
        event_labels_field = TextField([Token(lb) for lb in event_labels], token_indexers=self.cluster_name_indexer)
        participant_indices_field = MetadataField(participant_indices)
        participant_labels_field = TextField([Token(lb) for lb in participant_labels],
                                             token_indexers=self.cluster_name_indexer)
        merged_indices_field = MetadataField(merged_indices)
        merged_labels_field = TextField([Token(lb) for lb in merged_labels],
                                        token_indexers=self.cluster_name_indexer)
        scenario_field = TextField([Token(scenario)], token_indexers=self.scenario_indexer)
        # ([Token(t) for t in CONST.scenario_phrase_s[scenario].split()],
        #                              token_indexers=self.word_indexer)
        story_id_field = MetadataField(story_id)
        coreference_chain_idx_field =\
            MetadataField([int(chain_idx) if 'none' not in chain_idx else -1 for chain_idx in coreference_chain_idx])
        dep_marks_field = \
            MetadataField([[int(index) for index in dep_mark_inst.split(',')] if 'none' not in dep_mark_inst else [-1]
                           for dep_mark_inst in dep_marks])

        return Instance({'story': story_field,
                         'event_indices': event_indices_field,
                         'event_labels': event_labels_field,
                         'participant_indices': participant_indices_field,
                         'participant_labels': participant_labels_field,
                         'merged_indices': merged_indices_field,
                         'merged_labels': merged_labels_field,
                         'scenario': scenario_field,
                         'story_id': story_id_field,
                         'coreference_chain_idx': coreference_chain_idx_field,
                         'dep_marks': dep_marks_field})

    def _read(self, folder: str, exclude_protagonist=True) -> Iterable[Instance]:
        story_id = 0
        for scenario in os.listdir(folder):
            self.scenarios.append(scenario)
            # sometimes we need to deep copy the buffers when yielding them
            story_buffer = [f'{CONST.begin_of_story_type}_{scenario}']
            event_indices, event_labels = list(), list()
            participant_indices, participant_labels = list(), list()
            merged_indices, merged_labels = list(), list()
            coref_chain_idx = list()
            dep_marks = list()

            line_tuple = namedtuple(
                'instance', ['content', 'participant', 'part_head', 'event', 'event_head', 'coreference_chain_idx', 'dep_marks'])

            regularity_marks = list(open(os.path.join(self.regularity_prediction_folder, scenario), 'r')) \
                if self.regularity_prediction_folder is not None else None

            valid_lines = list()
            for line_index, line in enumerate(open(os.path.join(folder, scenario), 'r')):
                if (len(line) > 1 and '▁none none none none none none' not in line) \
                        or ('▁none none none none none none none' in line):
                    valid_lines.append(line)

            for line_index, line_str in enumerate(valid_lines):
                splited_line = line_str.split()
                story_buffer.append(splited_line[0])

                # end of story line, yield and clear buffers.
                if line_str.__contains__(CONST.end_of_story_type):
                    if (len(event_indices) != 0) and (len(participant_indices) != 0):
                        if not all(['evoking' in event_labels[i].lower() for i in range(len(event_labels))]):
                            if self.include_scenario_phrase:
                                scenario_phrase = CONST.all_scenario_phrases[scenario]
                                scenario_phrase_length = len(scenario_phrase)
                                story_buffer = scenario_phrase + story_buffer
                                event_indices = [ind + scenario_phrase_length for ind in event_indices]
                                participant_indices = [ind + scenario_phrase_length for ind in participant_indices]
                                merged_indices = [ind + scenario_phrase_length for ind in merged_indices]
                                #  event_labels = [CONST.null_label] * scenario_phrase_length + event_labels
                                #  participant_labels = [CONST.null_label] * scenario_phrase_length + participant_labels
                                #  merged_labels = [CONST.null_label] * scenario_phrase_length + merged_labels

                            assert len(merged_indices) == len(dep_marks)
                            assert len(participant_indices) == len(coref_chain_idx)
                            yield self.text_to_instance(story=story_buffer,
                                                        event_indices=event_indices,
                                                        event_labels=event_labels,
                                                        participant_indices=participant_indices,
                                                        participant_labels=participant_labels,
                                                        merged_indices=merged_indices,
                                                        merged_labels=merged_labels,
                                                        scenario=scenario,
                                                        story_id=story_id,
                                                        coreference_chain_idx=coref_chain_idx,
                                                        dep_marks=dep_marks)

                    story_id += 1
                    story_buffer = ['{}_{}'.format(CONST.begin_of_story_type, scenario)]
                    event_indices, event_labels = list(), list()
                    participant_indices, participant_labels = list(), list()
                    merged_indices, merged_labels = list(), list()
                    coref_chain_idx = list()
                    dep_marks = list()
                    continue

                # normal line
                elif len(splited_line) < 7:
                    line = line_tuple(
                        content=splited_line[0],
                        participant=splited_line[1],
                        part_head=splited_line[2],
                        event=splited_line[3],
                        event_head=splited_line[4],
                        coreference_chain_idx='none',
                        dep_marks='none'
                    )
                else:
                    line = line_tuple(
                        content=splited_line[0],
                        participant=splited_line[1],
                        part_head=splited_line[2],
                        event=splited_line[3],
                        event_head=splited_line[4],
                        coreference_chain_idx=splited_line[5],
                        dep_marks=splited_line[6]
                    )

                if line.coreference_chain_idx not in ['none', 'X']:
                    if int(line.coreference_chain_idx) % 100 == 0 and exclude_protagonist:
                        line = line_tuple(
                            content=splited_line[0],
                            participant='none',
                            part_head=splited_line[2],
                            event=splited_line[3],
                            event_head=splited_line[4],
                            coreference_chain_idx=splited_line[5],
                            dep_marks=splited_line[6]
                        )

                # event head
                if (line.event.lower() not in ['none', 'x'] and line.event_head not in ['none', 'no']) or \
                        ('evoking' in line.event.lower() and 'evoking' not in inscript_utils.irregular_prefixes['event']):
                    # valid event label
                    if self.regularity_prediction_folder is not None:
                        if '1' in regularity_marks[line_index] or ('evoking' in line.event.lower() and 'evoking' not in inscript_utils.irregular_prefixes['event']):
                            if self.include_irregular or inscript_utils.is_regular(modes=['event'], label=line.event):
                                event_indices.append(len(story_buffer) - 1)
                                event_labels.append(line.event)
                                merged_indices.append(len(story_buffer) - 1)
                                merged_labels.append(line.event)
                                dep_marks.append(line.dep_marks)
                                continue
                    elif self.include_irregular or inscript_utils.is_regular(modes=['event'], label=line.event):
                        event_indices.append(len(story_buffer) - 1)
                        event_labels.append(line.event)
                        merged_indices.append(len(story_buffer) - 1)
                        merged_labels.append(line.event)
                        dep_marks.append(line.dep_marks)
                        continue
                # participant head. In case a token receives both event and participant heads, only the event
                # annotation is kept
                if line.participant.lower() not in ['none', 'x'] and line.part_head not in ['none', 'no']:
                    # valid participant label
                    if self.regularity_prediction_folder is not None:
                        if '1' in regularity_marks[line_index]:
                            if self.include_irregular or inscript_utils.is_regular(modes=['participant'], label=line.participant):
                                participant_indices.append(len(story_buffer) - 1)
                                participant_labels.append(line.participant)
                                merged_indices.append(len(story_buffer) - 1)
                                merged_labels.append(line.participant)
                                coref_chain_idx.append(line.coreference_chain_idx)
                                dep_marks.append(line.dep_marks)
                    elif self.include_irregular or inscript_utils.is_regular(modes=['participant'], label=line.participant):
                        participant_indices.append(len(story_buffer) - 1)
                        participant_labels.append(line.participant)
                        merged_indices.append(len(story_buffer) - 1)
                        merged_labels.append(line.participant)
                        coref_chain_idx.append(line.coreference_chain_idx)
                        dep_marks.append(line.dep_marks)
        return

    @staticmethod
    def scenario_of_label(label: str):
        """
        return the scenario of a specific label
        """
        if '_' not in label:
            return '@UNK'
        else:
            return label[QualityOfPythonLife.last_index_of(label, '_') + 1:]

    def split(self, validation_scenario, test_scenario, data_instances, list_of_scenarios=None):
        if list_of_scenarios is None:
            list_of_scenarios = CONST.scenario_s

        if len(self.data) == 0:
            for scenario in list_of_scenarios:
                self.data[scenario] = \
                    [inst for inst in data_instances if inst.fields['scenario'].tokens[0].text == scenario]

        """ split data for the sake of cross-validation """
        train_scenarios = [scenario for scenario in self.scenarios
                           if scenario not in [test_scenario, validation_scenario]]
        train_data = AllennlpDataset(list(itertools.chain(*[self.data[scenario] for scenario in train_scenarios])))
        if validation_scenario != '':
            validation_data = AllennlpDataset(self.data[validation_scenario])
        else:
            validation_data = []
        if test_scenario != '':
            test_data = AllennlpDataset(self.data[test_scenario])
        else:
            test_data = []
        return train_data, validation_data, test_data


class ClusterInstanceSampler(BatchSampler):
    """
    a batch sampler that yields indices of a minibatch. Makes sure instances in each batch are from the same scenario.
    instances are shuffled.
    this BatchSampler requires a 'scenario' field in each instance. thus it works for both datasetReaders in this file.
    """

    def __init__(self, data_source: AllennlpDataset, batch_size: int, shuffle=False):
        """
        data_source: an AllennlpDataset object
        counts: dictionary giving counts of instances of each possible length
        :param data_source:
        """
        super(ClusterInstanceSampler, self).__init__()
        self.batch_size = batch_size
        self.indices_by_scenario = dict()
        self.num_batches_by_scenario = dict()
        self.data_source = data_source
        self.shuffle = shuffle
        for ind in range(len(data_source)):
            scenario = data_source[ind].fields['scenario'].tokens[0].text
            if scenario not in self.indices_by_scenario:
                self.indices_by_scenario[scenario] = [ind]
            else:
                self.indices_by_scenario[scenario].append(ind)

        self.scenarios = list(self.indices_by_scenario.keys())
        self.batch_details = dict()
        batch_ind = 0
        for scenario in self.indices_by_scenario:
            for i in range(math.ceil(len(self.indices_by_scenario[scenario]) / batch_size)):
                self.batch_details[batch_ind] = scenario, i * batch_size
                batch_ind += 1
        self.batch_indices = list(self.batch_details.keys())

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_indices)
            for scenario in self.indices_by_scenario:
                random.shuffle(self.indices_by_scenario[scenario])
        for ind in self.batch_indices:
            scenario, starting_idx = self.batch_details[ind]
            yield self.indices_by_scenario[scenario][starting_idx: starting_idx + self.batch_size]

    def __len__(self):
        return len(self.batch_details)


if __name__ == '__main__':
    pass

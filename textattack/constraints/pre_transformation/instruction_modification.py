"""

Stopword Modification
--------------------------

"""

import nltk

from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class InstructionModification(PreTransformationConstraint):
    """A constraint allowing moficiation of different parts of the input for models trained in instruction paradigm for LLM (e.g. GPT-3)"""

    def _get_word_index_range(self, main_string, to_search_string):
        word_index = 0
        start_char_index = main_string.find(to_search_string)
        end_char_index = start_char_index + len(to_search_string)
        for index, c in enumerate(main_string):
            if c == ' ':
                word_index += 1
            if index == start_char_index:
                start_index = word_index
            if index == end_char_index:
                end_index = word_index
                break
        return start_index, end_index

    def __init__(self, modify_definition=True, modify_task_input=True, modify_explanation=True):
        self.modify_definition = modify_definition
        self.modify_task_input = modify_task_input
        self.modify_explanation = modify_explanation

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        modifiable_indices = set()
        #words = current_text.words()
        new_text_input = current_text.text

        try:
            if self.modify_definition:
                definiton_str = new_text_input.split('Definition:')[1].split('Negative Examples:')[0]
                found_indexes = self._get_word_index_range(new_text_input, definiton_str)
                word_index_range = list(range(found_indexes[0], found_indexes[1]))
                modifiable_indices.update(word_index_range)
        except:
            print("No definition found in the text or error occured")
        try:
            if self.modify_task_input:
                task_input_str = new_text_input.split('Input:')[-1].split('Output:')[0]
                found_indexes = self._get_word_index_range(new_text_input, task_input_str)
                word_index_range = list(range(found_indexes[0], found_indexes[1]))
                modifiable_indices.update(word_index_range)
        except:
            print("No task input found in the text or error occured")
        try:
            if self.modify_explanation:
                text_input_parts = ''.join(new_text_input.split('Positive Examples:'))
                explanations = text_input_parts.split('Explanation:')
                last_explanation = explanations[-1].split('Input:')[0]
                explanation_list = explanations[1:-1] + [last_explanation]
                for exp in explanation_list:
                    found_indexes = self._get_word_index_range(new_text_input, exp)
                    word_index_range = list(range(found_indexes[0], found_indexes[1]))
                    modifiable_indices.update(word_index_range)
        except:
            print("No explanations found in the text or error occured")
        return modifiable_indices

    def check_compatibility(self, transformation):
        """
        not required
        """
        return True

from datasets import load_dataset
from typing import Dict, Any
import numpy as np
import io
import contextlib
import os
import pycodestyle


def score_lines(text: str) -> list:
    """
    Return list of PEP8 violations per character in each line of text.
    """
    virtual_file = io.StringIO(text)
    checker = pycodestyle.Checker(lines=virtual_file.readlines(), show_source=True)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):  # keep stdout clean
        try:
            _ = checker.check_all()
            scores = np.zeros(len(checker.lines))
            for line_number, offset, code, text, doc in checker.report._deferred_print:
                scores[line_number-1] += 1
            scores = scores/[len(line) for line in checker.lines]
        except (UnicodeEncodeError, ZeroDivisionError, IndexError):
            scores = np.zeros(len(checker.lines))  # this should be rare enough to not worry about
    return checker.lines(), scores.tolist(), len(checker.lines), np.mean(scores)


def score_element(element: Dict[str, Any]) -> Dict[str, Any]:
    element['texts'], element['scores'], element['num_lines'], element['avg_score'] = score_lines(element['text'])
    return element


dataset = load_dataset('codeparrot/codeparrot-train-more-filtering', split='train')
# subsample 1500k documents, should go beyond 3.3b tokens 
dataset = dataset.train_test_split(train_size=1500_000, shuffle=True)['train']

dataset = dataset.rename_column('content', 'org_text')
dataset = dataset.remove_columns(
    ['repo_name', 'path', 'copies', 'size', 'license', 'hash', 'line_mean', 'line_max', 'alpha_frac', 'autogenerated',
     'ratio', 'config_test', 'has_no_keywords', 'few_assignments']
)

print('Starting dataset scoring')
scored_dataset = dataset.map(score_element, num_proc=16)
print('Finished dataset scoring')
scored_dataset.push_to_hub('kejian/codeparrot-train-more-filtering-pep8-3.3b-scored')

# do filtering
scored_dataset = scored_dataset.map(lambda x: {'texts_match': ''.join(x['texts']) == x['org_text']}, num_proc=16)
scored_dataset = scored_dataset.filter(lambda x: x['texts_match'] is True)
scored_dataset = scored_dataset.remove_columns(['texts_match'])
scored_dataset.push_to_hub('kejian/codeparrot-train-more-filter-3.3b-cleaned')

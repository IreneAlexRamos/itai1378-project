"""Exercise real reporting logic without requiring local PyTorch installation."""
import ast
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gnasher_data import GROUPS

# Load only the pure NumPy reporting function from the actual module.
tree = ast.parse((ROOT / 'src/gnasher_training.py').read_text())
function = next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                and node.name=='summarize_predictions')
namespace = {'np':np,'GROUPS':GROUPS}
exec(compile(ast.Module(body=[function],type_ignores=[]),'<metrics>','exec'),namespace)
summarize = namespace['summarize_predictions']


class MetricAccountingTests(unittest.TestCase):
    def test_abstentions_remain_in_full_denominator(self):
        targets, probabilities = [], []
        for group in range(7):
            correct = np.full(7,.2/6)
            correct[group] = .8
            uncertain = np.full(7,.1)
            uncertain[(group+1)%7] = .4
            probabilities.extend([correct,uncertain])
            targets.extend([group,group])
        result = summarize(np.array(probabilities),np.array(targets),.50)
        self.assertEqual(result['n'],14)
        self.assertEqual(result['coverage'],.50)
        self.assertEqual(result['accepted_accuracy'],1.0)
        self.assertEqual(result['abstentions'],7)
        self.assertEqual(result['top1_accuracy'],.50)
        self.assertEqual(result['correct_with_abstentions_counted_in_denominator'],.50)
        decisions = np.array(result['decision_matrix_with_uncertain'])
        self.assertEqual(decisions.shape,(7,8))
        self.assertEqual(decisions.sum(),14)
        self.assertTrue((decisions[:,-1]==1).all())

    def test_all_uncertain_does_not_produce_nan_accepted_accuracy(self):
        result = summarize(np.zeros((7,7)),np.arange(7),.50)
        self.assertEqual(result['coverage'],0)
        self.assertIsNone(result['accepted_accuracy'])
        self.assertEqual(result['abstentions'],7)
        self.assertEqual(result['correct_with_abstentions_counted_in_denominator'],0)


if __name__=='__main__':
    unittest.main()

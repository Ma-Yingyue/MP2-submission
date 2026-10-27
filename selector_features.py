import ast
from collections import Counter
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer


_NUMBER = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_LEFT_EXPR = re.compile(r'([\d.()+*/^\- \t]+)$')
_RIGHT_NUMBER = re.compile(r'\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)')
_FRAC = re.compile(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}')
_STOP = set('a an and are as at be by for from had has have he her his how i in is it its many much of on or she that the their them then there they this to was were what when which who will with would you'.split())
_CUES = {
    'uncertainty': r'\b(?:assum\w*|perhaps|probably|maybe|might|seems?|suppose|approximately)\b',
    'revision': r'\b(?:however|instead|actually|mistake|incorrect|wrong|error|reconsider|wait)\b',
    'confidence': r'\b(?:clearly|obviously|definitely|certainly|therefore|thus|hence)\b',
    'verification': r'\b(?:check\w*|verif\w*|confirm\w*|consistent)\b',
    'conditions': r'\b(?:if|unless|except|including|remaining|left|after|before|half|twice|double|each|per)\b',
    'operations_words': r'\b(?:total|sum|difference|product|divide\w*|multiply\w*|subtract\w*|add\w*)\b',
}
_CUES = {key: re.compile(value) for key, value in _CUES.items()}


def _numeric(value):
    try:
        number = Fraction(value.replace(',', ''))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if number.numerator == 0:
        return 0.0, 0.0, 0.0, 1.0
    logarithm = math.log(abs(number.numerator)) - math.log(number.denominator)
    sign = 1.0 if number.numerator > 0 else -1.0
    bounded = sign * math.exp(min(logarithm, math.log(1e12))) if logarithm > -745 else 0.0
    return bounded, max(-40.0, min(40.0, logarithm)), sign, float(number.denominator == 1)


def _numbers(text):
    parsed = [_numeric(match.group()) for match in _NUMBER.finditer(text)]
    return [value[0] for value in parsed if value is not None]


def _close(left, right):
    return abs(left - right) <= 1e-7 * max(1.0, abs(left), abs(right))


def _safe_value(expression):
    if len(expression) > 100:
        raise ValueError
    tree = ast.parse(expression.replace('^', '**'), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand)
            if isinstance(node.op, ast.USub):
                result = -result
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.Pow) and abs(right) <= 6 and abs(left) <= 1e9:
                result = left ** right
            else:
                raise ValueError
        else:
            raise ValueError
        if not isinstance(result, (float, int)) or not math.isfinite(result) or abs(result) > 1e15:
            raise ValueError
        return result

    return visit(tree)


def _equations(text):
    normalized = text.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    normalized = normalized.replace('×', '*').replace('÷', '/').replace('−', '-')
    for _ in range(3):
        normalized = _FRAC.sub(r'((\1)/(\2))', normalized)
    normalized = re.sub(r'(?<=\d),(?=\d)', '', normalized)
    normalized = normalized.replace('$', '').replace('**', '').replace('\\left', '').replace('\\right', '')
    checked, correct, errors, last_correct, max_error = 0, 0, 0, 0.0, 0.0
    for match in list(re.finditer('=', normalized))[:80]:
        before = normalized[max(0, match.start() - 100):match.start()]
        left = _LEFT_EXPR.search(before)
        right = _RIGHT_NUMBER.match(normalized[match.end():match.end() + 100])
        if left is None or right is None:
            continue
        expression = left.group(1).strip().rstrip('.')
        if expression.startswith('- '):
            expression = expression[2:].lstrip()
        if not expression or not any(character in expression for character in '+-*/^'):
            continue
        try:
            actual = _safe_value(expression)
            expected_value = _numeric(right.group(1))
            if expected_value is None:
                continue
            expected = expected_value[0]
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
            continue
        checked += 1
        last_correct = float(_close(actual, expected))
        correct += int(last_correct)
        errors += int(not last_correct)
        max_error = max(max_error, min(20.0, math.log1p(abs(actual - expected) / max(1.0, abs(expected)))))
    return checked, correct, errors, correct / max(1, checked), last_correct, max_error


def majority_choices(bank):
    result = []
    for row in bank:
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts = Counter(answer for answer in answers if answer is not None)
        result.append(int(max(range(4), key=lambda index: (counts[answers[index]] if answers[index] is not None else -1, -index))))
    return result


def build_features(bank, arrays):
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (len(bank), 4):
        raise ValueError('Expected embeddings aligned to four candidates per question')
    if not np.isfinite(embeddings).all():
        raise ValueError('Expected finite embeddings')
    norms = np.linalg.norm(embeddings, axis=2)
    unit = embeddings / np.maximum(norms[..., None], 1e-12)
    similarities = np.einsum('nid,njd->nij', unit, unit)
    means = embeddings.mean(axis=1)
    mean_norms = np.linalg.norm(means, axis=1)
    mean_unit = means / np.maximum(mean_norms[:, None], 1e-12)
    mean_similarity = np.einsum('nid,nd->ni', unit, mean_unit)
    names = []
    output = []
    for group_index, row in enumerate(bank):
        question = row['question']
        question_lower = question.lower()
        question_words = _WORD.findall(question_lower)
        question_content = set(question_words) - _STOP
        question_numbers = _numbers(question)
        qn = np.asarray(question_numbers, dtype=np.float64)
        q_abs = np.abs(qn)
        q_nonzero = qn[q_abs > 1e-12]
        question_sum = float(np.clip(qn.sum(), -1e12, 1e12)) if len(qn) else 0.0
        question_product = 1.0
        for value in question_numbers:
            question_product = float(np.clip(question_product * value, -1e12, 1e12))
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts = Counter(answer for answer in answers if answer is not None)
        numeric_answers = [_numeric(answer) if answer is not None else None for answer in answers]
        numeric_valid = [value[0] for value in numeric_answers if value is not None]
        maximum_votes = max(counts.values(), default=0)
        top_answer_count = sum(value == maximum_votes for value in counts.values())
        group_features = []
        for candidate_index, candidate in enumerate(row['candidates']):
            text = candidate['text']
            lower = text.lower()
            words = _WORD.findall(lower)
            content = set(words) - _STOP
            trace_numbers = _numbers(text)
            trace_values = np.asarray(trace_numbers, dtype=np.float64)
            answer = answers[candidate_index]
            numeric = numeric_answers[candidate_index]
            valid = answer is not None
            votes = counts[answer] if valid else 0
            number, logarithm, sign, integer = numeric if numeric is not None else (0.0, 0.0, 0.0, 0.0)
            denominator = Fraction(answer).denominator if valid else 1
            overlap = len(question_content & content)
            same = [index for index in range(4) if index != candidate_index and answer is not None and answers[index] == answer]
            different = [index for index in range(4) if answers[index] is not None and answers[index] != answer]
            others = [index for index in range(4) if index != candidate_index]
            similarity = similarities[group_index, candidate_index]
            near_question = sum(any(_close(value, qvalue) for qvalue in question_numbers) for value in trace_numbers)
            answer_question_matches = sum(_close(number, value) for value in question_numbers) if valid else 0
            answer_trace_matches = sum(_close(number, value) for value in trace_numbers) if valid else 0
            answer_ratios = [number / value for value in q_nonzero] if valid else []
            inverse_ratios = [value / number for value in question_numbers] if valid and abs(number) > 1e-12 else []
            nearest_question = min((abs(number - value) / max(1.0, abs(number), abs(value)) for value in question_numbers), default=1.0) if valid else 1.0
            nearest_log = min((abs(math.log(max(1e-12, abs(number))) - math.log(max(1e-12, abs(value)))) for value in question_numbers), default=20.0) if valid else 20.0
            equation_checked, equation_correct, equation_errors, equation_rate, equation_last, equation_max_error = _equations(text)
            values = {
                'answer_valid': float(valid),
                'answer_votes': votes,
                'valid_candidates': sum(value is not None for value in answers),
                'distinct_valid_answers': len(counts),
                'maximum_answer_votes': maximum_votes,
                'answer_votes_gap': votes - maximum_votes,
                'answer_is_maximum_vote': float(valid and votes == maximum_votes),
                'maximum_vote_ties': top_answer_count,
                'answer_vote_rank': sum(value > votes for value in counts.values()),
                'answer_sign': sign,
                'answer_is_zero': float(valid and number == 0),
                'answer_is_integer': integer,
                'answer_log_abs': logarithm,
                'answer_signed_log': sign * math.log1p(abs(number)),
                'answer_abs_below_one': float(valid and 0 < abs(number) < 1),
                'answer_denominator_log': min(40.0, math.log(denominator)),
                'answer_numeric_rank': sum(value < number for value in numeric_valid) / max(1, len(numeric_valid)),
                'answer_group_log_median_gap': math.log1p(abs(number - float(np.median(numeric_valid)))) if valid and numeric_valid else 0.0,
                'answer_matches_question_number': answer_question_matches,
                'answer_matches_trace_number': answer_trace_matches,
                'answer_trace_number_fraction': answer_trace_matches / max(1, len(trace_numbers)),
                'answer_nearest_question_relative_gap': nearest_question,
                'answer_nearest_question_log_gap': min(40.0, nearest_log),
                'answer_ratio_question_integer_count': sum(abs(value - round(value)) < 1e-7 * max(1.0, abs(value)) for value in answer_ratios),
                'question_ratio_answer_integer_count': sum(abs(value - round(value)) < 1e-7 * max(1.0, abs(value)) for value in inverse_ratios),
                'answer_matches_question_sum': float(valid and len(qn) > 0 and _close(number, question_sum)),
                'answer_matches_question_product': float(valid and len(qn) > 0 and _close(number, question_product)),
                'answer_above_question_maximum': float(valid and len(qn) > 0 and number > float(qn.max())),
                'answer_below_question_minimum': float(valid and len(qn) > 0 and number < float(qn.min())),
                'question_characters': len(question),
                'question_words': len(question_words),
                'question_numbers': len(question_numbers),
                'question_maximum_abs_log': math.log1p(float(q_abs.max())) if len(qn) else 0.0,
                'question_minimum_abs_log': math.log1p(float(q_abs.min())) if len(qn) else 0.0,
                'question_has_percent': float('%' in question or 'percent' in question_lower),
                'question_has_money': float('$' in question or 'dollar' in question_lower),
                'trace_characters': len(text),
                'trace_words': len(words),
                'trace_lines': text.count('\n') + 1,
                'trace_unique_words': len(set(words)),
                'trace_content_words': len(content),
                'trace_word_diversity': len(set(words)) / max(1, len(words)),
                'question_word_recall': overlap / max(1, len(question_content)),
                'question_word_precision': overlap / max(1, len(content)),
                'question_word_jaccard': overlap / max(1, len(question_content | content)),
                'trace_numbers': len(trace_numbers),
                'trace_unique_numbers': len(set(trace_numbers)),
                'trace_question_number_overlap': near_question,
                'trace_question_number_fraction': near_question / max(1, len(trace_numbers)),
                'trace_new_number_fraction': (len(trace_numbers) - near_question) / max(1, len(trace_numbers)),
                'trace_number_word_fraction': len(trace_numbers) / max(1, len(words)),
                'trace_maximum_abs_number_log': math.log1p(float(np.max(np.abs(trace_values)))) if len(trace_values) else 0.0,
                'trace_equals': text.count('='),
                'trace_plus': text.count('+'),
                'trace_minus': text.count('-') + text.count('−'),
                'trace_multiply': text.count('\\times') + text.count('×') + text.count('*'),
                'trace_divide': text.count('\\frac') + text.count('\\div') + text.count('÷') + text.count('/'),
                'trace_hashes': text.count('#'),
                'trace_boxed': text.count('\\boxed'),
                'trace_bold': text.count('**'),
                'trace_latex': text.count('\\'),
                'trace_percent': text.count('%'),
                'trace_question_marks': text.count('?'),
                'trace_parentheses': text.count('('),
                'trace_bullets': len(re.findall(r'(?m)^\s*[-*]\s', text)),
                'trace_numbered_steps': len(re.findall(r'(?m)^\s*\d+[.)]\s', text)),
                'trace_final_answer_mentions': len(re.findall(r'\b(?:answer|result|conclusion)\b', lower)),
                'equation_checked': equation_checked,
                'equation_correct': equation_correct,
                'equation_errors': equation_errors,
                'equation_correct_rate': equation_rate,
                'equation_last_correct': equation_last,
                'equation_maximum_relative_error_log': equation_max_error,
                'embedding_norm': float(norms[group_index, candidate_index]),
                'embedding_norm_ratio': float(norms[group_index, candidate_index] / max(1e-12, norms[group_index].mean())),
                'embedding_cosine_group_mean': float(mean_similarity[group_index, candidate_index]),
                'embedding_cosine_others_mean': float(similarity[others].mean()),
                'embedding_cosine_others_minimum': float(similarity[others].min()),
                'embedding_cosine_others_maximum': float(similarity[others].max()),
                'embedding_cosine_others_std': float(similarity[others].std()),
                'embedding_cosine_same_answer_mean': float(similarity[same].mean()) if same else 0.0,
                'embedding_cosine_different_answer_mean': float(similarity[different].mean()) if different else 0.0,
                'embedding_cosine_answer_margin': (float(similarity[same].mean()) if same else 0.0) - (float(similarity[different].mean()) if different else 0.0),
                'embedding_group_mean_norm': float(mean_norms[group_index]),
            }
            for key, pattern in _CUES.items():
                count = len(pattern.findall(lower))
                values['trace_' + key] = count
                values['trace_' + key + '_rate'] = count / max(1, len(words))
            if not names:
                names = list(values)
            group_features.append([values[key] for key in names])
        output.append(group_features)
    if not bank:
        return np.empty((0, 4, 0), dtype=np.float32), []
    features = np.asarray(output, dtype=np.float32)
    relative_indices = [index for index, name in enumerate(names) if name.startswith(('trace_', 'equation_', 'embedding_')) and name != 'embedding_group_mean_norm']
    relative = features[:, :, relative_indices]
    maximum_gap = relative - relative.max(axis=1, keepdims=True)
    ranks = (relative[:, :, None, :] > relative[:, None, :, :]).sum(axis=2).astype(np.float32) / 3.0
    relative_names = [names[index] for index in relative_indices]
    grouped = {
        'answer_votes', 'valid_candidates', 'distinct_valid_answers', 'maximum_answer_votes',
        'answer_votes_gap', 'answer_is_maximum_vote', 'maximum_vote_ties', 'answer_vote_rank',
        'answer_numeric_rank', 'answer_group_log_median_gap',
    }
    names = [name if name.startswith('embedding_') else ('group_' if name in grouped else 'local_') + name for name in names]
    names = names + ['group_' + name + '_maximum_gap' for name in relative_names] + ['group_' + name + '_rank' for name in relative_names]
    features = np.concatenate((features, maximum_gap, ranks), axis=2)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=1e12, neginf=-1e12)
    return features, names

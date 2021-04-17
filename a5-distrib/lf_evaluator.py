import tempfile
import subprocess
import os
import re
from data import *

# YOU SHOULD NOT NEED TO LOOK AT THIS FILE.
# This file consists of evaluation code adapted from Jia + Liang, wrapping predictions and sending them to a Java
# backend for evaluation against the knowledge base.

def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None, use_java=True):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    if use_java:
        selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations, quiet=True)
    else:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    res = print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()
    return res

# Find the top-scoring derivation that executed without error
def pick_derivations(all_pred_dens, all_derivs, is_error_fn):
    derivs = []
    pred_dens = []
    cur_start = 0
    if len(all_pred_dens) == 0:
        print(len(all_derivs))
        print("No legal derivations! Likely you're getting an error when calling the evaluation in Java")
        for deriv_set in all_derivs:
            derivs.append(Derivation("", 0.0, [""]))
            pred_dens.append("Example FAILED TO PARSE")
        return (derivs, pred_dens)

    for deriv_set in all_derivs:
        # What to do if 0?
        for i in range(len(deriv_set)):
            cur_denotation = all_pred_dens[cur_start + i]
            if not is_error_fn(cur_denotation):
                derivs.append(deriv_set[i])
                pred_dens.append(cur_denotation)
                break
        else:
            if len(deriv_set) == 0:
                # Try to avoid crashing
                derivs.append(Derivation("", 0.0, [""]))
                pred_dens.append("Example FAILED TO PARSE")
            else:
                derivs.append(deriv_set[0])  # Default to first derivation
                pred_dens.append(all_pred_dens[cur_start])
        cur_start += len(deriv_set)
    return (derivs, pred_dens)


class GeoqueryDomain(object):
    def postprocess_lf(self, lf):
        # Undo the variable name standardization.
        cur_var = chr(ord('A') - 1)
        toks = lf.split(' ')
        new_toks = []
        for w in toks:
            if w == 'NV':
                cur_var = chr(ord(cur_var) + 1)
                new_toks.append(cur_var)
            elif w.startswith('V'):
                ind = int(w[1:])
                new_toks.append(chr(ord(cur_var) - ind))
            else:
                new_toks.append(w)
        return ' '.join(new_toks)

    def clean_name(self, name):
        return name.split(',')[0].replace("'", '').strip()

    def format_lf(self, lf):
        # Strip underscores, collapse spaces when not inside quotation marks
        lf = self.postprocess_lf(lf)
        toks = []
        in_quotes = False
        quoted_toks = []
        for t in lf.split():
            if in_quotes:
                if t == "'":
                    in_quotes = False
                    toks.append('"%s"' % ' '.join(quoted_toks))
                    quoted_toks = []
                else:
                    quoted_toks.append(t)
            else:
                if t == "'":
                    in_quotes = True
                else:
                    if len(t) > 1 and t.startswith('_'):
                        toks.append(t[1:])
                    else:
                        toks.append(t)
        lf = ''.join(toks)
        # Balance parentheses
        num_left_paren = sum(1 for c in lf if c == '(')
        num_right_paren = sum(1 for c in lf if c == ')')
        diff = num_left_paren - num_right_paren
        if diff > 0:
            lf = lf + ')' * diff
        return lf

    def get_denotation(self, line):
        m = re.search('\{[^}]*\}', line)
        if m:
            return m.group(0)
        else:
            return line.strip()

    def print_failures(self, dens, name):
        num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
        num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
        num_join_error = sum('Join failed syntactically' in d for d in dens)
        print('%s: %d syntax errors, %d executor errors' % (
            name, num_syntax_error, num_exec_error))

    def is_error(self, d):
        return 'FAILED' in d or 'Join failed syntactically' in d

    def compare_answers(self, true_answers, all_derivs, quiet=False):
        all_lfs = ([self.format_lf(s) for s in true_answers] +
                [self.format_lf(' '.join(d.y_toks))
                for x in all_derivs for d in x])
        tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
        tf = tempfile.NamedTemporaryFile(suffix='.dlog')
        for line in tf_lines:
            tf.write(line.encode() + b'\n')
            if not quiet:
                print(line)
        tf.flush()
        # JAVA INVOCATION: uncomment the following three lines to print the java code output and stop there if you
        # need to check if the Java is working
        #####
        # msg = subprocess.check_output(['evaluator/geoquery', tf.name], stderr=subprocess.STDOUT)
        # print(repr(msg.decode("utf-8"))
        # exit()
        #####
        try:
            msg = subprocess.check_output(['evaluator/geoquery', tf.name]).decode("utf-8")
            # Alternate form with the whole java command
            # msg = subprocess.check_output(['java', '-ea', '-server', '-Xss8m', '-cp', 'evaluator/evaluator.jar:lib/scala-compiler.jar:lib/scala-library.jar:lib/fig.jar:lib/tea.jar:lib/berkeleyParser.jar:lib/trove-2.1.0.jar',
            #                                'dcs.NuggetLearn', '-create', '-monitor', '-useStandardExecPoolDirStrategy', '-jarFiles', 'evaluator/evaluator.jar',
            #                                '+miscOptions', 'new4', '-model.verbose', '2', '-numIters', '5', '-updateType', 'full', '-miniBatchSize', 'MAX',
            #                                '-parser.command', '"bash lib/lowercase-parser"', '-parser.lowercase', 'true', '-useBayesianAveraging', 'true',
            #                                '-allowTroll', '-regularization', '0.01', '-beamSize', '100', '-features', 'pred', 'pred2', 'predarg', 'lexpred',
            #                                'lexnull', '-generalMaxExamples', 'MAX', '-data.permuteExamples', 'true', '-displayTypes', 'false', '-displayDens',
            #                                'false', '-displaySpans', 'false', '-displayMaxSetSize', '1', '-msPerLine', '0', '-int.verbose', '0', '-data.verbose',
            #                                '0', '-addToView', 'geo3', '-lexToName', '-lexToSetWithName', '-generalPaths', 'evaluator/domains/dbquery/geoquery/1/geoquery.dlog',
            #                                'evaluator/domains/dbquery/geoquery/1/lexicon.dlog', '-dlogOptions', 'lexMode=0', '+generalPaths', tf.name, '-trainFrac', '0.7',
            #                                '-testFrac', '0.3', '-data.random', '1'], stderr=subprocess.STDOUT).decode("utf-8")
            # Use this line instead if the subprocess call is crashing
            # msg = ""
        except subprocess.CalledProcessError as err:
            print("Error in subprocess Geoquery evaluation call. Command output:")
            print(err.output)
            print(err.returncode)
            print(msg)
            exit()
        tf.close()
        denotations = [self.get_denotation(line)
                       for line in msg.split('\n')
                       if line.startswith('        Example')]
        true_dens = denotations[:len(true_answers)]
        if len(true_dens) == 0:
            true_dens = ["" for i in range(0, len(true_answers))]
        all_pred_dens = denotations[len(true_answers):]

        # Find the top-scoring derivation that executed without error
        derivs, pred_dens = pick_derivations(all_pred_dens, all_derivs, self.is_error)
        if not quiet:
            self.print_failures(true_dens, 'gold')
            self.print_failures(pred_dens, 'predicted')
        for t, p in zip(true_dens, pred_dens):
            if not quiet:
                print('%s: %s == %s' % (t == p, t, p))
        return derivs, [t == p for t, p in zip(true_dens, pred_dens)]


##########################
# UNUSED IN THIS PROJECT #
##########################
# Evaluation code for the Overnight domains adapted from Robin Jia and Percy Liang.
class OvernightEvaluator(object):
    def format_lf(self, lf):
        replacements = [
            ('! ', '!'),
            ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
        ]
        for a, b in replacements:
            lf = lf.replace(a, b)
        # Balance parentheses
        num_left_paren = sum(1 for c in lf if c == '(')
        num_right_paren = sum(1 for c in lf if c == ')')
        diff = num_left_paren - num_right_paren
        if diff > 0:
            while len(lf) > 0 and lf[-1] == '(' and diff > 0:
                lf = lf[:-1]
                diff -= 1
            if len(lf) == 0: return ''
            lf = lf + ' )' * diff
        return lf

    def is_error(self, d):
        return 'BADJAVA' in d or 'ERROR' in d or d == 'null'

    def compare_answers(self, true_answers, all_derivs):
        # Put all "true" answers at the start of the list, then add all derivations that
        # were produced by decoding
        all_lfs = ([self.format_lf(s) for s in true_answers] +
                   [self.format_lf(' '.join(d.y_toks))
                    for x in all_derivs for d in x])
        tf_lines = all_lfs
        tf = tempfile.NamedTemporaryFile(suffix='.examples')
        for line in tf_lines:
            tf.write(line.encode() + b'\n')
            print(line)
        tf.flush()
        f = open(tf.name)
        subdomain = "calendar" # TODO: set subdomain
        msg = subprocess.check_output(['evaluator/overnight', subdomain, tf.name])
        tf.close()
        print(len(all_lfs))
        denotations = [line.split('\t')[1] for line in msg.decode("utf-8").split('\n')
                       if line.startswith('targetValue\t')]
        print(len(denotations))
        print(len(true_answers))
        true_dens = denotations[:len(true_answers)]
        all_pred_dens = denotations[len(true_answers):]
        derivs, pred_dens = pick_derivations(all_pred_dens, all_derivs, self.is_error)
        return derivs, [t == p for t, p in zip(true_dens, pred_dens)]

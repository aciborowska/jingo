# adapted from https://github.com/cscorley/triage
import logging
import sys
import string
import re
import numpy as np

logger = logging.getLogger('preprocessing')


def tokenize(s):
    return s.split()


def to_unicode(document, info=[]):
    if sys.version.startswith('3') and not isinstance(document, str):
        document = document.decode('utf-8', errors='ignore')
    document = document.replace('\x00', ' ')  # remove nulls
    document = document.replace('\r', '\n')
    document = document.replace('#', ' ')
    document = document.strip()
    if not isinstance(document, str):
        for codec in ['utf8', 'latin1', 'ascii']:
            try:
                return str(document, codec)
                # return unicode(document, encoding=codec)
            except UnicodeDecodeError as e:
                logger.debug('%s %s %s' % (codec, str(e), ' '.join(info)))

    return document


def split(iterator, keep_punctuation=False, preserve_code_tokens=True):
    for token in iterator:
        code_tokens = dict()
        nl_tokens = dict()
        word = u''
        code_token = u''
        for char in token:
            if char.isupper() and all(map(lambda x: x.isupper(), word)):
                # keep building if word is currently all uppercase
                word += char
                code_token += char
            elif char.islower() and all(map(lambda x: x.isupper(), word)):
                # stop building if word is currently all uppercase,
                # but be sure to take the first letter back
                # new version: emit splitted camel case but also preserve code token
                if len(word) > 1:
                    nl_tokens = increment(nl_tokens, word[:-1])
                    word = word[-1]
                word += char
                code_token += char
            elif char.islower() and any(map(lambda x: x.islower(), word)):
                # keep building if the word is has any lowercase
                # (word came from above case)
                word += char
                code_token += char
            elif char.isdigit() and all(map(lambda x: x.isdigit(), word)):
                # keep building if all of the word is a digit so far
                word += char
                code_token += char
            elif char in string.punctuation:
                if len(word) > 0:
                    nl_tokens = increment(nl_tokens, word)
                    code_tokens = increment(code_tokens, code_token)
                    code_token = u''
                    word = u''

                if keep_punctuation is True:
                    nl_tokens = increment(nl_tokens, char)

                # dont yield punctuation
                # yield char
            elif char == ' ':
                if len(word) > 0:
                    nl_tokens = increment(nl_tokens, word)
                    code_tokens = increment(code_tokens, code_token)

                word = u''
                code_token = u''
            else:
                if len(word) > 0:
                    nl_tokens = increment(nl_tokens, word)

                word = char
                code_token += char

        nl_tokens = increment(nl_tokens, word)
        code_tokens = increment(code_tokens, code_token)
        for t in nl_tokens:

            code_tokens[t] = 0
            for i in range(0, nl_tokens[t]):
                if len(t) > 0:
                    yield t

        if preserve_code_tokens is None:
            raise ValueError('Preserve_code_tokens is {0}'.format(preserve_code_tokens))

        if preserve_code_tokens is True:
            for t in code_tokens:
                for i in range(0, code_tokens[t]):
                    if len(t) > 0:
                        yield t


def increment(dictionary, key):
    if key not in dictionary:
        dictionary[key] = 0
    dictionary[key] += 1
    return dictionary


def remove_stops(iterator, stopwords=set(), punctuation=True, digits=True,
                 whitespace=True):
    if not isinstance(stopwords, set):
        stopwords = set(stopwords)

    if punctuation:
        stopwords.update(string.punctuation)

    if digits:
        stopwords.update(string.digits)

    if whitespace:
        stopwords.update(string.whitespace)

    stopwords.update([''])
    for word in filter(lambda x: x not in stopwords, iterator):
        try:
            int(word)
            float(word)
        except ValueError:
            yield word


def filter_code_snippets(text):
    lines = text.strip().split('\n')
    if len(lines) == 1:
        return _filter_1line_code_snippets(text)
    else:
        return _filter_multiline_text(lines)


def _filter_multiline_text(lines):
    islands = find_islands(lines)
    if len(islands) == 0:
        return '\n'.join(lines)

    code_snippet_lines = set(islands)
    verified = set(islands)

    while len(islands) > 0:
        line_idx = islands.pop()
        verified.add(line_idx)
        cs_lines = check_surroundings(lines, line_idx, verified, code_snippet_lines)
        if len(cs_lines) > 0:
            islands.extend(cs_lines)

    if len(code_snippet_lines) > 3:
        # keep 2 lines code snippets
        return '\n'.join(lines)

    nl_text = list()
    for idx, line in enumerate(lines):
        if idx not in code_snippet_lines and len(line) > 0:
            nl_text.append(line)

    return '\n'.join(nl_text)


def check_surroundings(lines, line_idx, verified, code_snippet_lines):
    start_idx = line_idx - 3 if len(lines) >= 3 else 0
    end_idx = line_idx + 3 if len(lines) > line_idx + 3 else len(lines)

    idx = start_idx - 1
    cs_lines = list()
    for line in lines[start_idx:end_idx]:
        idx += 1
        if idx in verified:
            continue
        verified.add(idx)
        if is_code(line) is True:
            code_snippet_lines.add(idx)
            cs_lines.append(idx)
    return cs_lines


def is_code(line):
    line = line.strip()
    if len(line) < 1:
        return False
    if line == '}' or line == '{':
        return True

    score = 0
    tokens = list(split(tokenize(line), keep_punctuation=True))

    if line.endswith(';'):
        score += len(tokens)

    for t in tokens:
        if t in ['(', ')', '{', '}', '=']:
            score += 1
        elif t in JAVA_RESERVED:
            score += 1
        elif camel_case(t) is True:
            score += 1

    probability = score / float(len(tokens) * 2)
    if probability > 0.6:
        return True
    return False


def camel_case(token):
    if token != token.lower() and token != token.upper() and token[0] == token[0].lower() and '/' not in token:
        return True
    return False


def find_islands(lines):
    class_re = re.compile(r'.*class [a-zA-Z0-9 ]+[ ]*{')
    import_re = re.compile(r'(package|import) [a-zA-Z0-9\.]+[*]{0,1};')
    assignment_re = re.compile(r'[a-zA-Z0-9 <>_\.,]+[ ]{0,1}=[ ]{0,1}(new ){0,1}[a-zA-Z0-9<> ,_\.\(\)\"]+;')
    method_re = re.compile(r'.* .*\(.*\).*{')

    islands = list()
    for idx, line in enumerate(lines):
        if class_re.match(line) or import_re.match(line) or assignment_re.match(line) or method_re.match(line):
            islands.append(idx)
    return islands


def filter_stackTrace(text, keep_top=3):
    st_regex = re.compile('at [a-zA-Z0-9\.<>$]*\(.+\)')
    words = list()
    kept = 0
    for line in text.split('\n'):
        if st_regex.match(line.strip()) is None:
            words.append(line)
        elif kept < keep_top:
            words.append(line)
            kept += 1

    return '\n'.join(words)


def _filter_1line_code_snippets(text):
    tokens = split_code(text)
    suspicious = np.zeros(len(tokens))
    method_inv = re.compile(r'[a-zA-Z0-9\.]+\(.*\)')
    for idx, t in enumerate(tokens):
        if re.match(method_inv, t) and not t.endswith(';'):
            suspicious[idx] = 0
        elif t == '-' or t == ',' or t == '/' or t == ':':
            suspicious[idx] = 0
        elif t.endswith(';'):
            suspicious[idx] = 1
        elif any(ext in t for ext in ['{', '}', '.', '(', ')']) is True:
            suspicious[idx] = 1
        elif ';' in t or '/**' == t or t == '*' or t == '*/' or t == '()' or t == '&':
            suspicious[idx] = 1
        elif t in JAVA_RESERVED:
            suspicious[idx] = 1
        elif camel_case(t) is True:
            suspicious[idx] = 1

    # merge neighbor islands
    islands = list()
    start_idx = -1
    end_idx = -1
    for idx, val in enumerate(suspicious):
        if val == 1:
            if start_idx == -1:
                start_idx = idx
            end_idx = idx
        elif val == 0:
            if start_idx == -1:
                continue
            else:
                islands.append((start_idx, end_idx))
                start_idx = -1
                end_idx = -1
    if start_idx != -1:
        islands.append((start_idx, end_idx))

    # merge islands
    idx = 0
    while (idx < len(islands)):
        next_idx = idx + 1
        while (merge_island(islands, idx, next_idx) is True):
            del islands[next_idx]
        idx += 1

    # filter
    nl_text = list()
    cs_tokens = islands2set(islands)
    for idx, t in enumerate(tokens):
        if idx not in cs_tokens:
            nl_text.append(t)

    return ' '.join(nl_text)


def merge_island(islands, current_idx, next_idx):
    if next_idx >= len(islands):
        return False
    start1, end1 = islands[current_idx]
    start2, end2 = islands[next_idx]

    if start2 - end1 <= 1:
        islands[current_idx] = (start1, end2)
        return True
    elif (start2 - end1) <= 3 and (end2 - start2 >= 3 or end1 - start1 >= 3):
        islands[current_idx] = (start1, end2)
    elif (start2 - end1) <= 5 and (end2 - start2 >= 3 and end1 - start1 >= 3):
        islands[current_idx] = (start1, end2)
        return True

    return False


def islands2set(islands, threshold=5):
    indices = set()
    for island in islands:
        start, end = island
        if end - start > threshold:
            indices.update([x for x in range(start, end + 1)])
    return indices


def code_tokens_in_window(idx, suspicious, window=5):
    if suspicious[idx] == 1.0:
        return 1.0
    start_idx = idx - window if idx - window > 0 else 0
    end_idx = idx + window + 1 if len(suspicious) > idx + window + 1 else len(suspicious)

    ssum = 0
    for sidx, val in enumerate(suspicious[start_idx:end_idx]):
        ssum += val

    return ssum / float(end_idx - start_idx)


def split_code(text):
    text = text.strip()
    tokens = list()
    word = ''
    method = False

    for char in text:
        if char.isalpha() or char.isdigit():
            word += char
        elif char.isdigit() and all(map(lambda x: x.isdigit(), word)):
            word += char
        elif char == '.' or char == ';' or char == '$':
            if len(word) > 0:
                word += char
        elif char == '(':
            method = True
            word += char
        elif char == ')' and method is True:
            method = False
            word += char
        elif char in [',', '<', '<', ' '] and method is True:
            word += char
        elif char == ' ' and method is True:
            method = False
            if len(word) > 0:
                tokens.append(word)
            word == ''
        elif char == ' ':
            if len(word) > 0:
                tokens.append(word)
            word = ''
        elif char == '{' or char == '}':
            if len(word) > 0:
                tokens.append(word)
            tokens.append(char)
            word = ''
        elif char not in ['\'']:
            if len(word) > 0:
                tokens.append(word)
            tokens.append(char)
            word = ''

    return tokens


FOX_STOPS = set(
    """ a about above across after again against all almost alone along already
    also although always among an and another any anybody anyone anything
    anywhere are area areas around as ask asked asking asks at away b back
    backed backing backs be because become becomes became been before began
    behind being beings best better between big both but by c came can cannot
    case cases certain certainly clear clearly come could d did differ different
    differently do does done down downed downing downs during e each early
    either end ended ending ends enough even evenly ever every everybody
    everyone everything everywhere f face faces fact facts far felt few find
    finds first for four from full fully further furthered furthering furthers
    g gave general generally get gets give given gives go going good goods got
    great greater greatest group grouped grouping groups h had has have having
    he her herself here high higher highest him himself his how however i if
    important in interest interested interesting interests into is it its itself
    j just k keep keeps kind knew know known knows l large largely last later
    latest least less let lets like likely long longer longest m made make
    making man many may me member members men might more most mostly mr mrs much
    must my myself n necessary need needed needing needs never new newer newest
    next no non not nobody noone nothing now nowhere number numbered numbering
    numbers o of off often old older oldest on once one only open opened opening
    opens or order ordered ordering orders other others our out over p part
    parted parting parts per perhaps place places point pointed pointing points
    possible present presented presenting presents problem problems put puts
    q quite r rather really right room rooms s said same saw say says second
    seconds see sees seem seemed seeming seems several shall she should show
    showed showing shows side sides since small smaller smallest so some
    somebody someone something somewhere state states still such sure t take
    taken than that the their them then there therefore these they thing things
    think thinks this those though thought thoughts three through thus to today
    together too took toward turn turned turning turns two u under until up upon
    us use uses used v very w want wanted wanting wants was way ways we well
    wells went were what when where whether which while who whole whose why will
    with within without work worked working works would x y year years yet you
    young younger youngest your yours z """.split())

JAVA_RESERVED = set(
    """ abstract assert boolean break byte case catch char class const continue
    default do double else enum extends false final finally float for goto if
    implements import instanceof int interface long native new null package
    private protected public return short static strictfp super switch
    synchronized this throw throws transient true try void volatile while """.split())

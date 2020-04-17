import common
import preprocessing
import re
import os
import git

import utils


def count(project, bug, classes_list, filter_cs=True):
    bug_report = common.read_issue(project, bug, filter_cs=filter_cs)
    code_tokens = extract_code_tokens(bug_report, classes_list)

    words_no = 0
    bug_report_tok = bug_report.split(' ')
    bug_report_tok = [token.replace('//', '').strip() for token in bug_report_tok]
    bug_report_tok = [token.replace('\"', '') for token in bug_report_tok]
    bug_report_tok = [token.replace(';', '') for token in bug_report_tok]
    bug_report_tok = [token.replace('=', '') for token in bug_report_tok]
    for word in bug_report_tok:
        if word not in code_tokens and word not in preprocessing.FOX_STOPS and word != '':
            words_no += 1

    tokens_no = len(code_tokens)

    print('########## {0} ###########\n{1}\n{2}\nwords = {3}, tokens = {4}\n'.format(bug, bug_report, code_tokens,
                                                                                     words_no, tokens_no))
    return words_no, tokens_no


def extract_code_tokens(bug, classes_list):
    classnames = extract_classnames(bug, classes_list)
    camel_case_regex = r'((?:[a-zA-Z])(?:\S?)+(?:[A-Z])(?:[a-z])+|([a-zA-Z]+\(([a-zA-Z]*)\))|([a-zA-Z]+\.[a-zA-Z\.]+)|([a-zA-Z]+(_[a-zA-Z]+)+))'
    code_tokens = re.findall(camel_case_regex, bug)
    code_tokens = [x[0] for x in code_tokens]
    for word in bug.split(' '):
        if word in preprocessing.JAVA_RESERVED and word not in preprocessing.FOX_STOPS:
            code_tokens.append(word)

    code_tokens += classnames
    return code_tokens


def extract_classnames(bug, classes_list=None):
    classname_regex = r'(?:([ \.$(][a-zA-Z][A-Za-z0-9]+))|(?:(^[A-Z][A-Za-z0-9]+))'

    names = re.findall(classname_regex, bug)
    names = [name for name_tuple in names for name in name_tuple]
    names = [name.replace(' ', '') for name in names]
    names = [name.replace('.', '') for name in names]
    names = [name.replace('$', '') for name in names]
    names = [x for x in names if len(x.strip()) > 3]

    classnames = []
    classes_list = [x.lower() for x in classes_list]
    if classes_list:
        # if classes list is not None, then use it to verify that extracted names are in fact classes
        for name in names:
            if name.lower() in classes_list:
                classnames.append(name.lower())
    else:
        # return extracted names and hope they are actually classes
        classnames = names

    return classnames


def list_source_code_files(project, file_extensions=['.java']):
    repo_path = clone_repo(project)

    classes = []
    for root, dirs, files in os.walk(repo_path):
        for name in files:
            if any(name.endswith(extension) for extension in file_extensions):
                classname = name.split('/')[-1][:-5]
                classes.append(classname)

    return classes


def clone_repo(project, repo_path='repos'):
    if not os.path.exists(os.path.join(repo_path, project.name)):
        with open(os.path.join(project.datasets, project.name, 'repos.txt')) as f:
            repo_urls = [line.strip() for line in f]

        utils.mkdir(repo_path)
        git.Git(repo_path).clone(repo_urls[0])

    repo_path = os.path.join(repo_path, project.name)
    return repo_path

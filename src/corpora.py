# extended from https://github.com/cscorley/triage

import io
import re
import sys
import os
import os.path
import javalang
import logging

import gensim
import dulwich
import dulwich.objects
import dulwich.index
import dulwich.repo
import dulwich.patch

import preprocessing
from gensim.parsing.porter import PorterStemmer

logger = logging.getLogger('corpora')


class GeneralCorpus(gensim.interfaces.CorpusABC):
    def __init__(self, project=None, id2word=None, remove_stops=True, split=True, lower=True,
                 min_len=3, max_len=40, lazy_dict=False, label='general'):
        lazystr = str()
        if lazy_dict:
            lazystr = 'lazy '

        logger.info('Creating %s%s corpus', lazystr, self.__class__.__name__)

        self.project = project
        self.remove_stops = remove_stops
        self.split = split
        self.lower = lower
        self.min_len = min_len
        self.max_len = max_len
        self.lazy_dict = lazy_dict
        self.dict_provided = False
        if self.project is not None:
            self.preserve_code_tokens = True if project.model_type == 'joined' else False
        else:
            self.preserve_code_tokens = None

        if id2word is None:
            id2word = gensim.corpora.Dictionary()
            logger.debug('[gen] Creating new dictionary %s for %s %s',
                         id(id2word), self.__class__.__name__, id(self))
        else:
            logger.debug('[gen] Using provided dictionary %s for %s %s',
                         id(id2word), self.__class__.__name__, id(self))
            self.dict_provided = True

        self.id2word = id2word

        self.metadata = False
        self.label = label

        if not lazy_dict and not self.dict_provided:
            # build the dict (not lazy)
            self.id2word.add_documents(self.get_texts())

        super(GeneralCorpus, self).__init__()

    @property
    def id2word(self):
        return self._id2word

    @id2word.setter
    def id2word(self, val):
        logger.debug('[gen] Updating dictionary %s for %s %s', id(val),
                     self.__class__.__name__, id(self))
        self._id2word = val

    def preprocess(self, document, info=[]):
        document = preprocessing.to_unicode(document, info)
        words = preprocessing.tokenize(document)

        if self.split:
            words = preprocessing.split(words, preserve_code_tokens=self.preserve_code_tokens)

        if self.lower:
            words = (word.lower() for word in words)

        if self.remove_stops:
            words = preprocessing.remove_stops(words, preprocessing.FOX_STOPS)
            words = preprocessing.remove_stops(words, preprocessing.JAVA_RESERVED)

        def include(word):
            return len(word) >= self.min_len and len(word) <= self.max_len

        p = PorterStemmer()
        words = [p.stem(word) for word in words if include(word)]
        return words

    def __iter__(self):
        """
        The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each
        document.
        """
        for text in self.get_texts():
            if self.metadata:
                meta = text[1]
                text = text[0]
                yield (self.id2word.doc2bow(text, allow_update=self.lazy_dict),
                       meta)
            else:
                yield self.id2word.doc2bow(text, allow_update=self.lazy_dict)

    def __len__(self):
        return self.length  # will throw if corpus not initialized


class GitCorpus(GeneralCorpus):
    def __init__(self, repo, project=None, remove_stops=True, split=True,
                 lower=True, min_len=3, max_len=40, id2word=None,
                 lazy_dict=False, label=None, ref=None, divide_commits=False):

        self.divide_commits = divide_commits

        if ref is None:
            if project and project.ref:
                ref = project.ref
            else:
                ref = 'HEAD'

        logger.debug('[git] Creating %s corpus out of source files for commit %s: %s',
                     self.__class__.__name__, ref, str(lazy_dict))
        self.repo = repo

        # ensure ref is a str otherwise dulwich cries
        if isinstance(ref, str):
            self.ref = ref.encode('utf-8')
        else:
            self.ref = ref

        self.ref_tree = None
        self.ref_commit_sha = None

        # find which file tree is for the commit we care about
        try:
            if sys.version.startswith('2'):
                self.ref_obj = self.repo[self.ref.encode('utf-8')]
            else:
                self.ref_obj = self.repo[self.ref]
        except:
            raise ValueError('Could not find ref %s in repo, using HEAD', self.ref)
            logger.info('Could not find ref %s in repo, using HEAD', self.ref)
            self.ref_obj = self.repo[self.repo.head()]

        if isinstance(self.ref_obj, dulwich.objects.Tag):
            self.ref_tree = self.repo[self.ref_obj.object[1]].tree
            self.ref_commit_sha = self.ref_obj.object[1]
        elif isinstance(self.ref_obj, dulwich.objects.Commit):
            self.ref_tree = self.ref_obj.tree
            self.ref_commit_sha = self.ref_obj.id
        elif isinstance(self.ref_obj, dulwich.objects.Tree):
            self.ref_tree = self.ref_obj.id
        else:
            self.ref_tree = self.ref  # here goes nothin?

        # set the label
        # filter to get rid of all empty strings
        if label is None:
            label = list(filter(lambda x: x, repo.path.split('/')))[-1]

        super(GitCorpus, self).__init__(project=project,
                                        remove_stops=remove_stops,
                                        split=split,
                                        lower=lower,
                                        min_len=min_len,
                                        max_len=max_len,
                                        id2word=id2word,
                                        lazy_dict=lazy_dict,
                                        label=label)


class MethodCorpus(GitCorpus):

    def get_texts(self):
        length = 0

        for entry in self.repo.object_store.iter_tree_contents(self.ref_tree):
            fname = entry.path.decode('utf-8')
            if not fname.endswith(".java") and not fname.endswith('.kt'):
                continue

            document = self.repo.object_store.get_raw(entry.sha)[1]

            if dulwich.patch.is_binary(document) or document is None or len(document) == 0:
                continue

            document = preprocessing.to_unicode(document, [str(fname), self.ref])
            try:
                tree = javalang.parse.parse(document)
                for path, node in tree.filter(javalang.tree.MethodDeclaration):
                    words = self._visit(node, [])

                    if self.project.model_type == 'joined':
                        classname = fname.split('/')[-1][:-5]
                        for i in range(0, 10):
                            words.append(classname.lower())

                        # add filename
                        for t in fname.replace('.', '/').split('/')[:-1]:
                            words.append(t.lower())

                    words = self.preprocess(' '.join(words), [str(fname), self.ref])
                    length += 1

                    if self.metadata:
                        yield words, (str(fname), self.label)
                    else:
                        yield words
            except:
                # cannot extract methods so let's leave full class file
                logger.info('Cannot process file {0}'.format(fname))
                words = self.preprocess(document, [str(fname), self.ref])

                # increase classname weight
                if self.project.model_type == 'joined':
                    classname = fname.split('/')[-1][:-5]
                    for i in range(0, 10):
                        words.append(classname.lower())
                    # add filename
                    for t in fname.replace('.', '/').split('/')[:-1]:
                        words.append(t.lower())

                length += 1

                if self.metadata:
                    yield words, (str(fname), self.label)
                else:
                    yield words
                continue

        self.length = length

    def _visit(self, root, words):
        if not isinstance(root, javalang.tree.Node):
            # if root is not Node it's either: str, list, set or None
            if isinstance(root, str):
                words.append(root)
                return words
            if root is None:
                return words
            else:
                try:
                    for val in root:
                        words = self._visit(val, words)
                except TypeError:
                    # object is not iterable, too bad
                    pass
        else:
            for attr in root.attrs:
                words = self._visit(getattr(root, attr), words)

        return words


class ReleaseCorpus(GeneralCorpus):
    def __init__(self, project, remove_stops=True, split=True, lower=True,
                 min_len=3, max_len=40, id2word=None, lazy_dict=False, label='release'):

        self.src = project.src_path
        super(ReleaseCorpus, self).__init__(project=project,
                                            remove_stops=remove_stops,
                                            split=split,
                                            lower=lower,
                                            min_len=min_len,
                                            max_len=max_len,
                                            id2word=id2word,
                                            label=label)

    def get_texts(self):
        length = 0
        for dirpath, dirnames, filenames in os.walk(self.src):
            if '.git' in dirnames:
                dirnames.remove('.git')
            for fname in filenames:
                path = os.path.join(dirpath, fname)
                with open(path) as f:
                    document = f.read()

                # lets not read binary files :)
                if dulwich.patch.is_binary(document):
                    continue

                path = path[len(self.src):]

                words = self.preprocess(document, [path, self.src])
                length += 1

                if self.metadata:
                    yield words, (path, self.label)
                else:
                    yield words

        self.length = length  # only reset after iteration is done.


class SnapshotSegmentCorpus(GitCorpus):
    def get_texts(self):
        length = 0

        for entry in self.repo.object_store.iter_tree_contents(self.ref_tree):
            fname = entry.path.decode('utf-8')
            if not fname.endswith(".java") and not fname.endswith('.kt'):
                continue

            document = self.repo.object_store.get_raw(entry.sha)[1]

            if dulwich.patch.is_binary(document):
                continue

            words = self.preprocess(document, [str(fname), self.ref])
            segment_size = 10
            segments_cnt = int((len(words) + segment_size - 1) / segment_size)

            for cnt in range(0, segments_cnt):
                start_idx = cnt * segment_size
                end_idx = (cnt + 1) * segment_size
                words_seg = words[start_idx:end_idx]

                # increase classname weight
                if self.project.model_type == 'joined':
                    classname = fname.split('/')[-1][:-5]
                    for i in range(0, 10):
                        words_seg.append(classname.lower())
                    # add filename
                    for t in fname.replace('.', '/').split('/')[:-1]:
                        words_seg.append(t.lower())

                length += 1

                if self.metadata:
                    yield words_seg, (str(fname), self.label)
                else:
                    yield words_seg

        self.length = length  # only reset after iteration is done.


class SnapshotCorpus(GitCorpus):
    def get_texts(self):
        length = 0

        for entry in self.repo.object_store.iter_tree_contents(self.ref_tree):
            fname = entry.path.decode('utf-8')
            if not fname.endswith(".java") and not fname.endswith('.kt'):
                continue

            document = self.repo.object_store.get_raw(entry.sha)[1]

            if dulwich.patch.is_binary(document):
                continue

            words = self.preprocess(document, [str(fname), self.ref])

            # increase classname weight
            if self.project.model_type == 'joined':
                classname = fname.split('/')[-1][:-5]
                for i in range(0, 10):
                    words.append(classname.lower())
                # add filename
                for t in fname.replace('.', '/').split('/')[:-1]:
                    words.append(t.lower())

            length += 1

            if self.metadata:
                yield words, (str(fname), self.label)
            else:
                yield words

        self.length = length  # only reset after iteration is done.


class ChangesetCorpus(GitCorpus):
    def __init__(self, repo, project=None, remove_stops=True, split=True,
                 lower=True, min_len=3, max_len=40, id2word=None,
                 lazy_dict=False, label=None, ref=None, include_removals=True,
                 include_additions=True, include_context=True,
                 include_message=False, include_filenames=True, divide_commits=False):
        self.include_removals = include_removals
        self.include_additions = include_additions
        self.include_context = include_context
        self.include_message = include_message
        self.include_filenames = include_filenames
        self.timestamps = dict()
        self.commit_msgs = dict()
        super(ChangesetCorpus, self).__init__(repo,
                                              project=project,
                                              remove_stops=remove_stops,
                                              split=split,
                                              lower=lower,
                                              min_len=min_len,
                                              max_len=max_len,
                                              id2word=id2word,
                                              lazy_dict=lazy_dict,
                                              label=label,
                                              ref=ref,
                                              divide_commits=divide_commits)

    def _get_diff(self, changeset):
        """ Return a text representing a `git diff` for the files in the
        changeset.

        """
        patch_file = io.BytesIO()
        try:
            dulwich.patch.write_object_diff(patch_file,
                                            self.repo.object_store,
                                            changeset.old, changeset.new)
        except UnicodeDecodeError as e:
            logger.debug(e)
            return b''

        return patch_file.getvalue()

    def _walk_changes(self):
        """ Returns one file change at a time, not the entire diff.

        """

        for walk_entry in self.repo.get_walker(include=[self.ref_commit_sha], reverse=True):
            commit = walk_entry.commit

            # initial revision, has no parent
            if len(commit.parents) == 0:
                for changes in dulwich.diff_tree.tree_changes(
                        self.repo.object_store, None, commit.tree):
                    diff = self._get_diff(changes)
                    yield commit, None, diff

            for parent in commit.parents:
                # do I need to know the parent id?

                for changes in dulwich.diff_tree.tree_changes(
                        self.repo.object_store, self.repo[parent].tree, commit.tree):
                    diff = self._get_diff(changes)
                    yield commit, parent, diff

    def get_texts(self):
        length = 0
        unified = re.compile(r'^[+ -].*')
        context = re.compile(r'^ .*')
        addition = re.compile(r'^\+.*')
        removal = re.compile(r'^-.*')
        no_source_code = r'.*(\.xml)|(\.groovy)|(\.sh)|(\.properties)|(LICENSE)|(LICENSE\.txt)|(NOTICE\.txt)|(.md)|(\.html)'
        current = None
        low = list()  # collecting the list of words

        for commit, parent, diff in self._walk_changes():
            # write out once all diff lines for commit have been collected
            # this is over all parents and all files of the commit
            diff = diff.decode("utf-8", errors='ignore')
            if current is None:
                # set current for the first commit, clear low
                current = commit.id.decode("utf-8")
                commit_msg = commit.message.decode("utf-8", errors='ignore').replace('\n', ' ')
                current_timestamp = commit._commit_time
                low = list()
            elif current != commit.id.decode("utf-8") and not self.divide_commits:
                # new commit seen, yield the collected low
                if self.metadata:
                    yield low, (current, self.label)
                else:
                    yield low

                self.timestamps[current] = current_timestamp
                self.commit_msgs[current] = commit_msg
                length += 1
                current = commit.id.decode("utf-8")
                commit_msg = commit.message.decode("utf-8", errors='ignore').replace('\n', ' ')
                current_timestamp = commit._commit_time
                low = list()
            elif self.divide_commits:
                current = commit.id.id.decode("utf-8")
                commit_msg = commit.message.decode("utf-8", errors='ignore').replace('\n', ' ')
                current_timestamp = commit._commit_time

            # to process out whitespace only changes, the rest of this
            # loop will need to be structured differently. possibly need
            # to actually parse the diff to gain structure knowledge
            # (ie, line numbers of the changes).

            diff_lines = list(filter(lambda x: unified.match(x), diff.splitlines()))

            if len(diff_lines) < 2:
                continue  # useful for not worrying with binary files

            # if len(re.findall(no_source_code, diff_lines[1])) > 0:
            #    continue

            # discard non *.java files
            if re.compile(r'.*\.java').match(diff_lines[1]) is None and re.compile(r'.*\.kt').match(diff_lines[1]) is None:
                continue

            # sanity?
            assert diff_lines[0].startswith('--- '), diff_lines[0]
            assert diff_lines[1].startswith('+++ '), diff_lines[1]
            # parent_fn = diff_lines[0][4:]
            # commit_fn = diff_lines[1][4:]

            classname1 = diff_lines[0].replace('--- ', '').replace('/dev/null', '').split('/')[-1][:-5]
            classname2 = diff_lines[1].replace('+++ ', '').replace('/dev/null', '').split('/')[-1][:-5]
            filename1 = diff_lines[0].replace('--- ', '').replace('/dev/null', '')
            filename2 = diff_lines[1].replace('+++', '').replace('/dev/null', '')

            lines = diff_lines[2:]  # chop off file names hashtag rebel

            if not self.include_additions:
                lines = filter(lambda x: not addition.match(x), lines)

            if not self.include_removals:
                lines = filter(lambda x: not removal.match(x), lines)

            if not self.include_context:
                lines = filter(lambda x: not context.match(x), lines)

            lines = [line[1:] for line in lines]  # remove unified markers

            if self.include_message:
                lines.append(commit.message.decode("utf-8", errors='ignore'))

            document = ' '.join(lines)

            # do the preprocessing steps
            if parent is not None:
                parent = parent.decode("utf-8")

            words = self.preprocess(document,
                                    [commit.id.decode("utf-8"), parent, diff_lines[0]])

            if self.project.model_type == 'joined':
                words.extend(self._names(classname2, filename2))

            if self.divide_commits:
                length += 1
                self.timestamps[current] = current_timestamp
                self.commit_msgs[current] = commit_msg
                if self.metadata:
                    # have reached the end, yield whatever was collected last
                    yield words, (current, self.label)
                else:
                    yield words

            low.extend(words)

        if not self.divide_commits:
            length += 1
            self.timestamps[current] = current_timestamp
            self.commit_msgs[current] = commit_msg
            if self.metadata:
                yield low, (current, self.label)
            else:
                yield low

        self.length = length  # only reset after iteration is done.

    def _names(self, classname2, filename2, cn_weight=10):
        tokens = list()
        for i in range(0, cn_weight):
            tokens.append(classname2.lower())

        for t in filename2.replace('.', '/').split('/')[:-1]:
            tokens.append(t.lower())

        return tokens


class CommitLogCorpus(GitCorpus):

    def get_texts(self):
        length = 0
        source_code = re.compile(r'.*\.java')
        unified = re.compile(r'^[+ -].*')

        for walk_entry in self.repo.get_walker(include=[self.ref_commit_sha], reverse=True):
            commit = walk_entry.commit
            changed_files = False
            if len(commit.parents) == 0:

                for changes in dulwich.diff_tree.tree_changes(self.repo.object_store, None, commit.tree):
                    if self.divide_commits:
                        diff = self._get_diff(changes)
                        diff_lines = filter(lambda x: unified.match(x), diff.splitlines())
                        if len(diff_lines) < 2:
                            continue  # useful for not worrying with binary files

                        # if source_code.match(diff_lines[1]) is None:
                        #     #print(diff_lines[1])
                        #     continue

                        words = self.preprocess(commit.message, [commit.id])
                        if self.metadata:
                            yield words, (commit.id, self.label)
                        else:
                            yield words
                        length += 1
                    else:
                        changed_files = True
            else:
                for parent in commit.parents:
                    # do I need to know the parent id?
                    for changes in dulwich.diff_tree.tree_changes(
                            self.repo.object_store, self.repo[parent].tree, commit.tree):
                        if self.divide_commits:
                            diff = self._get_diff(changes)
                            diff_lines = filter(lambda x: unified.match(x), diff.splitlines())
                            if len(diff_lines) < 2:
                                continue  # useful for not worrying with binary files

                            # if source_code.match(diff_lines[1]) is None:
                            #     #print(diff_lines[1])
                            #     continue

                            words = self.preprocess(commit.message, [commit.id])
                            if self.metadata:
                                yield words, (commit.id, self.label)
                            else:
                                yield words
                            length += 1
                        else:
                            changed_files = True
            if not changed_files:
                continue

            if not self.divide_commits:
                words = self.preprocess(commit.message, [commit.id])

                length += 1

                if self.metadata:
                    yield words, (commit.id, self.label)
                else:
                    yield words

        self.length = length  # only reset after iteration is done.

    def _get_diff(self, changeset):
        """ Return a text representing a `git diff` for the files in the
        changeset.

        """
        patch_file = io.StringIO()
        dulwich.patch.write_object_diff(patch_file,
                                        self.repo.object_store,
                                        changeset.old, changeset.new)
        return patch_file.getvalue()


class CorpusCombiner(GeneralCorpus):
    def __init__(self, corpora=None, id2word=None):
        self.corpora = list()

        super(CorpusCombiner, self).__init__(id2word=id2word, lazy_dict=True)

        if corpora:
            for each in corpora:
                self.add(each)

    def add(self, corpus):
        trans = self.id2word.merge_with(corpus.id2word)
        corpus.metadata = self.metadata
        corpus.id2word = self.id2word
        corpus.word2id = self.id2word.token2id
        self.corpora.append(corpus)

    def __iter__(self):
        for corpus in self.corpora:
            for doc in corpus:
                yield doc

    def __len__(self):
        return sum(len(c) for c in self.corpora)

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, val):
        assert val is True or val is False
        self._metadata = val
        for corpus in self.corpora:
            corpus.metadata = self._metadata

    @property
    def id2word(self):
        return self._id2word

    @id2word.setter
    def id2word(self, val):
        logger.debug('[com] Updating dictionary %s for %s %s', id(val),
                     self.__class__.__name__, id(self))
        self._id2word = val
        for corpus in self.corpora:
            corpus.id2word = val

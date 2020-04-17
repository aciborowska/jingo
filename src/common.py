# extended from https://github.com/cscorley/triage
import logging

import csv
import os
import os.path
import operator
import pickle
from collections import namedtuple

import dulwich.repo

from gensim.corpora import MalletCorpus, Dictionary
from gensim.parsing.porter import PorterStemmer
from gensim.matutils import sparse2full
from gensim.utils import smart_open, to_unicode, to_utf8

import utils
from corpora import (ChangesetCorpus, SnapshotCorpus, CorpusCombiner, GeneralCorpus, CommitLogCorpus, MethodCorpus)
import preprocessing as pp

logger = logging.getLogger('common')


def write_ranks(project, prefix, ranks, path=None):
    if path is None:
        path = os.path.join(project.full_path, '-'.join([prefix, project.level, 'ranks.csv.gz']))
    else:
        path = os.path.join(path, '-'.join([prefix, project.level, 'ranks.csv.gz']))
    logger.info("Attempting to write %d ranks to: %s", len(ranks), path)
    with smart_open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'rank', 'distance', 'item'])

        for gid, rank in ranks.items():
            for idx, dist, d_name in rank:
                writer.writerow([gid, idx, dist, to_utf8(d_name)])


def read_ranks(project, prefix):
    path = os.path.join(project.full_path, '-'.join([prefix, project.level, 'ranks.csv.gz']))
    logger.info("Attempting to read ranks from: %s", path)
    ranks = dict()
    with smart_open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for g_id, idx, dist, d_name in reader:
            if g_id not in ranks:
                ranks[g_id] = list()

            ranks[g_id].append((int(idx), float(dist), to_unicode(d_name)))

    logger.info("Read %d ranks", len(ranks))

    return ranks


def get_frms(ranks, goldsets):
    logger.info('Getting FRMS for %d ranks', len(ranks))
    frms = list()

    for r_id, rank in ranks.items():
        if r_id not in goldsets:
            logger.info('Skipping %s, not in goldset', str(r_id))
            continue

        added = False
        for idx, dist, name in rank:
            if name in goldsets[r_id]:
                added = True
                frms.append((idx, r_id, name))
                break  # take only the first one

        if not added:
            logger.info('Found no FRM for %s goldset \n\t %s', str(r_id), str(goldsets[r_id]))

    logger.info('Returning %d FRMS', len(frms))
    return frms


def get_rels(ranks, goldset=None):
    rels = list()

    if goldset:
        dists = list()
        for name in goldset:
            if name in ranks:
                dists.append((ranks[name], name))

        for dist, name in dists:
            idx = len([1 for x in ranks.values() if x < dist])
            rels.append((idx + 1, dist, name))

    else:
        # without the goldset, we have to sort and enumerate all items :(
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
        for idx, rank in enumerate(sorted_ranks):
            d_name, dist = rank
            rels.append((idx + 1, dist, d_name))

    rels.sort()

    return rels


def get_rank(project, query_topic, doc_topic, goldsets=None, distance_measure=utils.hellinger_distance):
    logger.info('Getting ranks between %d query topics and %d doc topics', len(query_topic), len(doc_topic))
    if project.level == 'file':
        return _get_rank_file(query_topic, doc_topic, goldsets, distance_measure)
    elif project.level == 'method':
        return _get_rank_method(query_topic, doc_topic, goldsets, distance_measure)
    else:
        raise ValueError('Unknown level {0}.'.format(project.level))


def _get_rank_file(query_topic, doc_topic, goldsets, distance_measure):
    ranks = dict()
    for q_meta, query in query_topic:
        qid, _ = q_meta
        q_dist = dict()
        # vectors = dict()

        for d_meta, doc in doc_topic:
            d_name, _ = d_meta
            q_dist[d_name] = distance_measure(query, doc)
            # vectors[d_name] = (query, doc)

        if goldsets and qid in goldsets:
            goldset = goldsets[qid]
        else:
            goldset = None

        sorted_dist = sorted(q_dist.items(), key=operator.itemgetter(1))
        ranks[qid] = get_rels(q_dist, goldset)

    logger.info('Returning %d ranks', len(ranks))
    return ranks, sorted_dist


def _get_rank_method(query_topic, doc_topic, goldsets, distance_measure):
    ranks = dict()
    for q_meta, query in query_topic:
        qid, _ = q_meta
        q_dist = dict()

        for d_meta, doc in doc_topic:
            d_name, _ = d_meta
            distance = distance_measure(query, doc)
            if d_name not in q_dist:
                q_dist[d_name] = distance
            elif q_dist[d_name] > distance:
                q_dist[d_name] = distance

        if goldsets and qid in goldsets:
            goldset = goldsets[qid]
        else:
            goldset = None

        sorted_dist = sorted(q_dist.items(), key=operator.itemgetter(1))
        ranks[qid] = get_rels(q_dist, goldset)

    logger.info('Returning %d ranks', len(ranks))
    return ranks, sorted_dist


def get_topics(model, corpus, by_ids=None, full=True):
    logger.info('Getting doc topic for corpus with length %d, by ids %s', len(corpus), str(by_ids))
    doc_topic = list()
    corpus.metadata = True
    old_id2word = corpus.id2word
    corpus.id2word = model.id2word

    if by_ids:
        by_ids = set(by_ids)
        by_ids.update([str(x) for x in by_ids])
    logger.debug("BYIDS:%s", by_ids)

    for doc, metadata in corpus:
        logger.debug("METADATA:%s", str(metadata))
        if by_ids is None or metadata[0] in by_ids:
            # get a vector where low topic values are zeroed out.
            topics = model[doc]
            if full:
                topics = sparse2full(topics, model.num_topics)

            doc_topic.append((metadata, topics))

    corpus.metadata = False
    corpus.id2word = old_id2word
    logger.info('Returning doc topic of length %d', len(doc_topic))

    return doc_topic


def load_ids(project):
    with open(os.path.join(project.full_path, 'ids.txt')) as f:
        ids = [x.strip() for x in f.readlines()]

    return ids


def load_fixed_ids(project, version_ids=True):
    if version_ids:
        path = os.path.join(project.full_path, 'fixed_ids.txt')
        with open(path) as f:
            ids = [x.strip() for x in f.readlines()]
    else:
        ids = list()
        for dir_name in get_subdirs(project.data_path):
            if dir_name.startswith('v'):
                fname = os.path.join(project.data_path, dir_name, 'fixed_ids.txt')
                if os.path.exists(fname):
                    with open(fname, 'r') as f:
                        ids_v = [x.strip() for x in f.readlines()]
                    ids += ids_v
    return ids


def get_subdirs(root):
    return [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]


def load_bug_timestamps(project, all_bugs=True):
    if all_bugs is True:
        path = os.path.join(project.data_path, 'bug_ids.txt')
    else:
        path = os.path.join(project.full_path, 'affected_ids.txt')

    with open(path) as f:
        ids = [x.strip().split(',') for x in f.readlines()]
        ids = [(x[0], int(float(x[1]))) for x in ids]
    return ids


def load_issue2git(project, ids, filter_ids=False):
    logger.info("Loading issue2git.csv")
    dest_fn = os.path.join(project.data_path, 'issue2git.csv')
    if os.path.exists(dest_fn):
        write_out = False
        i2g = dict()
        with open(dest_fn) as f:
            r = csv.reader(f)
            for issue, repo, sha in r:
                if issue not in i2g:
                    i2g[issue] = list()
                i2g[issue].append(sha)

    else:
        write_out = True

        i2s = dict()
        fn = os.path.join(project.full_path, 'IssuesToSVNCommitsMapping.txt')
        with open(fn) as f:
            lines = [line.strip().split('\t') for line in f]
            for line in lines:
                issue = line[0]
                links = line[1]
                svns = line[2:]

                i2s[issue] = svns

        s2g = dict()
        fn = os.path.join(project.data_path, 'svn2git.csv')
        with open(fn) as f:
            reader = csv.reader(f)
            for svn, git in reader:
                if svn in s2g and s2g[svn] != git:
                    logger.info('Different gits sha for SVN revision number %s', svn)
                else:
                    s2g[svn] = git

        i2g = dict()
        for issue, svns in i2s.items():
            for svn in svns:
                if svn in s2g:
                    # make sure we don't have issues that are empty
                    if issue not in i2g:
                        i2g[issue] = list()
                    i2g[issue].append(s2g[svn])
                else:
                    logger.info('Could not find git sha for SVN revision number %s', svn)

    logger.info("Loaded issue2git with %d entries", len(i2g))

    # Make sure we have a commit for all issues
    keys = set(i2g.keys())
    ignore = set(ids) - keys
    if len(ignore):
        logger.info("Ignoring evaluation for the following issues:\n\t%s",
                    '\n\t'.join(ignore))

    # clean up by ids if needed:
    if filter_ids:
        for issue in list(i2g.keys()):
            if issue not in ids:
                del i2g[issue]

    # build reverse mapping
    g2i = dict()
    for issue, gits in i2g.items():
        for git in gits:
            if git not in g2i:
                g2i[git] = list()
            g2i[git].append(issue)

    if write_out:
        with open(dest_fn, 'w') as f:
            w = csv.writer(f)
            for issue, gits in i2g.items():
                w.writerow([issue] + gits)

    logger.info("Returning issue2git with len %d and git2issue with len %d", len(i2g), len(g2i))

    return i2g, g2i


def load_projects(config, path='datasets'):
    projects = list()
    refpaths = list()
    for dirpath, dirname, filenames in os.walk(path):
        for filename in filenames:
            if filename == 'ref':
                refpaths.append(os.path.join(dirpath, filename))

    for refpath in refpaths:
        with open(refpath) as f:
            ref = f.read().strip()

        full_path, _ = os.path.split(refpath)
        data_path, project_version = os.path.split(full_path)
        _, project_name = os.path.split(data_path)
        src_path = os.path.join(full_path, 'src')

        path = '{0}/changeset_{1}_model_{8}/bugs_{2}_changes_{3}_decays_{4}_{5}/gamma_{6}_omega_{7}' \
            .format(project_name,
                    config['changeset_config_string'],
                    config['bugs_model_config']['num_topics'],
                    config['changes_model_config']['num_topics'],
                    config['bugs_model_config']['decay'],
                    config['changes_model_config']['decay'],
                    config['gamma'],
                    config['omega'],
                    config['model_type']
                    )

        config['results_path'] = os.path.join('results', path)
        config['save_model_path'] = os.path.join('models', path)
        config['level'] = 'method' if config['model_type'] == 'joined' else 'file'

        Project = namedtuple('Project',
                             ' '.join(['name', 'printable_name', 'version', 'ref', 'data_path', 'full_path', 'src_path']
                                      + list(config.keys())))

        row = [project_name, make_title(project_name) + ' ' + project_version, project_version, ref,
               os.path.join(data_path, ''),
               os.path.join(full_path, ''),
               os.path.join(src_path, '')]
        row += config.values()

        projects.append(Project(*row))

    return projects


def make_title(name):
    return name.title().replace("keeper", "Keeper").replace("jpa", "JPA")


def load_repos(project):
    # reading in repos
    with open(os.path.join(project.datasets, project.name, 'repos.txt')) as f:
        repo_urls = [line.strip() for line in f]

    repos_base = 'gits'
    if not os.path.exists(repos_base):
        utils.mkdir(repos_base)

    repos = list()

    for url in repo_urls:
        repo_name = url.split('/')[-1]
        target = os.path.join(repos_base, repo_name)
        try:
            repo = utils.clone(url, target, bare=True)
        except OSError:
            repo = dulwich.repo.Repo(target)

        repos.append(repo)

    return repos


def create_queries(project, id2word=None):
    corpus_fname_base = 'Queries'
    corpus_fname = corpus_fname_base + '.mallet.gz'
    dict_fname = corpus_fname_base + '.dict.gz'
    corpus_save_path = os.path.join(project.save_model_path, corpus_fname)
    dict_save_path = os.path.join(project.save_model_path, dict_fname)

    if not os.path.exists(corpus_save_path):
        pp = GeneralCorpus(lazy_dict=True, project=project)
        if id2word is None:
            id2word = Dictionary()

        bug_timestamps = load_bug_timestamps(project)
        queries = _build_queries_corpus(project, bug_timestamps, pp, id2word, allow_update=True)

        MalletCorpus.serialize(corpus_save_path, queries, id2word=id2word, metadata=True)
        id2word.save(dict_save_path)

    # re-open the compressed versions of the dictionary and corpus
    id2word = None
    if os.path.exists(dict_save_path):
        id2word = Dictionary.load(dict_save_path)

    corpus = MalletCorpus(corpus_save_path, id2word=id2word)

    return corpus


def _build_queries_corpus(project, bug_timestamps, pp, id2word, allow_update):
    queries = list()
    prev_timestamp = 0
    for bugid, timestamp in bug_timestamps:
        # bugs should be sorted by creation time
        assert int(timestamp) >= prev_timestamp
        text = read_issue(project, bugid)
        text = pp.preprocess(text)

        bow = id2word.doc2bow(text, allow_update)
        queries.append((bow, (bugid, 'query')))
        prev_timestamp = int(timestamp)
    return queries


def read_issue(project, id, join=True, filter_cs=False):
    with open(os.path.join(project.data_path, 'queries',
                           'short', '%s.txt' % id)) as f:
        short = f.read()

    with open(os.path.join(project.data_path, 'queries',
                           'long', '%s.txt' % id)) as f:
        long = f.read()

    if filter_cs is True:
        long = pp.filter_code_snippets(long)

    if join:
        return ' '.join([short, long])
    else:
        return short, long


def create_corpus(project, repos, Kind, use_level=True, forced_ref=None, mallet=True):
    names = [Kind.__name__]
    args = {
        'project': project,
        'lazy_dict': True,
    }
    if mallet is False:
        args['lazy_dict'] = False

    if use_level:
        names.append(project.level)

    if Kind is ChangesetCorpus:
        names.append(project.changeset_config_string)
        args.update(project.changeset_config)
    elif Kind is CommitLogCorpus:
        names.append(project.commit_logs_config_string)
        args.update(project.commit_logs_config)

    if forced_ref:
        names.append(str(forced_ref[:8]))

    corpus_fname_base = '-'.join(names)

    corpus_fname = corpus_fname_base + '.mallet.gz'
    dict_fname = corpus_fname_base + '.dict.gz'
    timestamps_fname = corpus_fname + '_timestamps.pkl'
    commit_msgs = corpus_fname + '_commitlog.pkl'
    corpus_save_path = os.path.join(project.save_model_path, corpus_fname)
    dict_save_path = os.path.join(project.save_model_path, dict_fname)
    timestamps_save_path = os.path.join(project.save_model_path, timestamps_fname)
    commit_msgs_save_path = os.path.join(project.save_model_path, commit_msgs)

    if not os.path.exists(corpus_save_path):
        combiner = CorpusCombiner()

        for repo in repos:
            try:
                if repo or forced_ref:
                    args.update({
                        'repo': repo,
                        'ref': forced_ref,
                    })
                corpus = Kind(**args)

            except KeyError:
                continue

            combiner.add(corpus)
            made_one = True

        # write the corpus and dictionary to disk. this will take awhile.
        combiner.metadata = True

        if mallet is True:
            MalletCorpus.serialize(corpus_save_path, combiner, id2word=combiner.id2word,
                                   metadata=True)
            combiner.metadata = False

            if hasattr(corpus, 'timestamps'):
                f = open(timestamps_save_path, 'wb')
                pickle.dump(corpus.timestamps, f)
                f.close()

            if hasattr(corpus, 'commit_msgs'):
                f = open(commit_msgs_save_path, 'wb')
                pickle.dump(corpus.commit_msgs, f)
                f.close()
            # write out the dictionary
            combiner.id2word.save(dict_save_path)
        else:
            if Kind is ChangesetCorpus:
                return corpus, corpus.timestamps, corpus.commit_msgs
            else:
                return corpus

    # re-open the compressed versions of the dictionary and corpus
    id2word = None
    if os.path.exists(dict_save_path):
        id2word = Dictionary.load(dict_save_path)

    corpus = MalletCorpus(corpus_save_path, id2word=id2word)

    if Kind is ChangesetCorpus:
        f = open(timestamps_save_path, 'rb')
        timestamps = pickle.load(f)
        f.close()
        f = open(commit_msgs_save_path, 'rb')
        commit_msgs = pickle.load(f)
        f.close()
    else:
        timestamps = None
        commit_msgs = None

    return corpus, timestamps, commit_msgs


def create_release_corpus(project, repos, ref=None, mallet=True):
    if project.level == 'file':
        SC = SnapshotCorpus
    elif project.level == 'method':
        SC = MethodCorpus
    else:
        raise ValueError('Unknown values for level = {0}'.format(project.level))

    return create_corpus(project, repos, SC, forced_ref=ref, mallet=mallet)


def preprocess(document, preserve_code_tokens, info=[]):
    document = pp.to_unicode(document, info)
    words = pp.tokenize(document)

    words = pp.split(words, preserve_code_tokens=preserve_code_tokens)

    words = (word.lower() for word in words)

    words = pp.remove_stops(words, pp.FOX_STOPS)
    words = pp.remove_stops(words, pp.JAVA_RESERVED)

    def include(word):
        return len(word) >= 3 and len(word) <= 40

    p = PorterStemmer()
    words = [p.stem(word) for word in words if include(word)]
    return words


def issue2bow(project, bug_corpus, bug_id):
    preserve_ct = True if project.model_type == 'joined' else False
    issue_raw_text = read_issue(project, bug_id)
    issue_text = preprocess(issue_raw_text, preserve_code_tokens=preserve_ct)
    issue_bow = bug_corpus.id2word.doc2bow(issue_text, allow_update=False)
    return issue_bow

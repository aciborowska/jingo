import logging
import numpy as np
import common

logger = logging.getLogger('common')


def generate_links(project, c_model, b_model, version_ids=True):
    # link format:
    # (prev_commit_idx, idx_fix_start, idx_fix_end, sha, commit_ts, prev_bug_idx, end_bug_idx, fixed_bugs)
    logger.info('Generate fixed links according to git2issues links')

    changesets_corpus = c_model.changeset_corpus

    links = list()
    sha_list = set()
    prev_commit_pos = 0
    prev_bug_pos = 0
    commit_fix_sha = None

    ids = common.load_fixed_ids(project, version_ids)
    issue2git, git2issue = common.load_issue2git(project, ids, filter_ids=True)

    bug_timestamps = common.load_bug_timestamps(project)
    bugs_id2pos = id2position(b_model.bugs_corpus, bug_timestamps)

    changesets_corpus.metadata = True

    for commit_pos, docmeta in enumerate(changesets_corpus):
        doc, meta = docmeta
        sha, _ = meta

        if commit_fix_sha is not None and sha != commit_fix_sha:
            # new commit arrived (different sha)
            # create link based on previous sha
            fixed_bugs = np.unique(git2issue[commit_fix_sha])
            commit_ts = c_model.changeset_ts[commit_fix_sha]
            bug_id = int(get_first_bug_after_timestamp(bug_timestamps, commit_ts))
            bug_pos = bugs_id2pos[bug_id][0]
            end_fix = commit_pos - 1

            links.append((prev_commit_pos, start_fix_pos, end_fix, commit_fix_sha, commit_ts, prev_bug_pos, bug_pos,
                          fixed_bugs))

            prev_commit_pos = end_fix
            prev_bug_pos = bug_pos
            commit_fix_sha = None

        if sha in git2issue and sha not in sha_list:
            start_fix_pos = commit_pos
            commit_fix_sha = sha
            sha_list.add(sha)

    changesets_corpus.metadata = False
    logger.info('Created {0} partitions'.format(len(links)))
    return links


def id2position(bug_corpus, bugs_ts):
    id2position = dict()
    bug_corpus.metadata = True

    for corpus_idx, docmeta in enumerate(bug_corpus):
        doc, meta = docmeta
        bug_id_corpus, _ = meta

        for bug_id, ts in bugs_ts:
            if int(bug_id_corpus) == int(bug_id):
                id2position[int(bug_id)] = (corpus_idx, bug_id, ts)
                break

    bug_corpus.metadata = False

    # make sure that all bugs were mapped
    assert len(id2position) == len(bugs_ts)
    return id2position


def get_first_bug_after_timestamp(bug_timestamps, timestamp):
    for bug_ts in bug_timestamps:
        bug_id, ts = bug_ts
        if int(ts) > timestamp:
            return bug_id
    return bug_id


def generate_links_by_timestamps(links, timestamp, last_fixed_link_idx=None):
    # (prev_commit_idx, idx_fix_start, idx_fix_end, sha, commit_ts, prev_bug_idx, end_bug_idx, fixed_bugs)
    seen_fixes = list()

    # if we have starting point let's use it
    if last_fixed_link_idx is None:
        last_fixed_link_idx = 0
    else:
        for link in links[:last_fixed_link_idx]:
            seen_fixes.append(link)

    for link in links[last_fixed_link_idx:]:
        commit_ts = link[4]
        if commit_ts < timestamp:
            seen_fixes.append(link)

    return seen_fixes


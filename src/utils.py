# from https://github.com/cscorley/triage
import os
import sys
import numpy
import scipy
import scipy.spatial
import dulwich.client
import logging

logger = logging.getLogger('utils')

SQRT2 = numpy.sqrt(2)


def calculate_mrr(p):
    vals = list()
    for idx, item in enumerate(p):
        if item:
            vals.append(1.0 / float(item))
        else:
            logger.info('penalty added for index %d', idx)
            vals.append(0.0)

    return numpy.mean(vals)


def calculate_map(ranks):
    ap = list()
    seen = set()
    for bug_id, recl in ranks:
        vals = list()
        if bug_id in seen:
            continue
        seen.add(bug_id)
        for idx, file_ranks in enumerate(recl):
            rank, _, _ = file_ranks
            vals.append((idx + 1) / float(rank))
        if len(recl) > 0:
            ap.append(numpy.mean(vals))

    return numpy.mean(ap)


def calculate_top(ranking, k=10):
    cnt = 0
    for bug in ranking:
        for file_ranks in ranking[bug]:
            rank, _, _ = file_ranks
            if rank <= k:
                cnt += 1
                break
            elif rank > k:
                break

    return cnt / float(len(ranking))


def hellinger_distance(p, q):
    p = rescale(p)
    q = numpy.abs(numpy.array(q))
    return scipy.linalg.norm(numpy.sqrt(p) - numpy.sqrt(q)) / SQRT2


def kullback_leibler_divergence(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return scipy.stats.entropy(p, q)


def cosine_distance(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return scipy.spatial.distance.cosine(p, q)


def jensen_shannon_divergence(p, q):
    p = rescale(p)
    q = numpy.array(q)
    M = (p + q) / 2
    return (kullback_leibler_divergence(p, M) +
            kullback_leibler_divergence(p, M)) / 2


def norm(p):
    return [float(i) / sum(p) for i in p]


def rescale(p):
    p = numpy.array(p)
    if min(p) < 0:
        min_p = abs(min(p))
        for i in range(0, len(p)):
            p[i] += min_p
    sum_p = sum(p)
    for i in range(0, len(p)):
        p[i] = p[i] / float(sum_p)
    return p


def total_variation_distance(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return numpy.sum(numpy.abs(p - q)) / 2


def score(model, fn):
    # thomas et al 2011 msr
    scores = list()
    for a, topic_a in norm_phi(model):
        score = 0.0
        for b, topic_b in norm_phi(model):
            if a == b:
                continue

            score += fn(topic_a, topic_b)

        score *= (1.0 / (model.num_topics - 1))
        logger.debug("topic %d score %f" % (a, score))
        scores.append((a, score))

    return scores


def norm_phi(model):
    for topicid in range(model.num_topics):
        topic = model.state.get_lambda()[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        yield topicid, topic


def mkdir(d):
    # exception handling mkdir -p
    try:
        os.makedirs(d)
    except os.error as e:
        if 17 == e.errno:
            # the directory already exists
            pass
        else:
            print('Failed to create "%s" directory!' % d)
            sys.exit(e.errno)


def download_file(url, destdir):
    # modified from http://stackoverflow.com/a/16696317
    # delay import until now
    import requests
    local_filename = os.path.join(destdir, url.split('/')[-1])
    if not os.path.exists(local_filename):
        # NOTE the stream=True parameter
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
    return local_filename


def clone(source, target, bare=False):
    client, host_path = dulwich.client.get_transport_and_path(source)

    if target is None:
        target = host_path.split("/")[-1]

    if not os.path.exists(target):
        os.mkdir(target)

    if bare:
        r = dulwich.repo.Repo.init_bare(target)
    else:
        r = dulwich.repo.Repo.init(target)

    remote_refs = client.fetch(host_path, r, determine_wants=r.object_store.determine_wants_all)

    r["HEAD".encode('utf-8')] = remote_refs["HEAD".encode('utf-8')]

    for key, val in remote_refs.items():
        if not key.endswith('^{}'.encode('utf-8')):
            r.refs.add_if_new(key, val)

    return r

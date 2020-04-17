import os
import numpy as np

import pyLDAvis.gensim

import utils
import common


def changeset_prediction(project, bug_id, c_model, code_topics, bug_topics, subranks, all_ranks):
    target_files = [x[2] for x in subranks]
    rec_files = [x[0] for x in all_ranks[:10]]
    path = os.path.join(project.results_path, 'bug_prediction')
    utils.mkdir(path)
    issue = common.read_issue(project, bug_id)

    bug_topics = list(zip(range(0, project.topics[1]), bug_topics.reshape(-1)))
    bug_topics.sort(key=lambda x: x[1], reverse=True)

    preserve_ct = True if project.model_type == 'joined' else False
    with(open(os.path.join(path, '{0}.txt'.format(bug_id)), 'a')) as f:
        f.write('Issue {0}\n'.format(bug_id))
        f.write(issue)

        f.write('\n############# Bug topics ################')
        top_topics = 5 if len(bug_topics) >= 5 else len(bug_topics)
        for topic, prob in bug_topics[:top_topics]:
            f.write('\nTopics {0} - {1} : {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))

        f.write('\n#########################')
        f.write('\nProbability of terms in the top topics')
        issue_text = common.preprocess(issue, preserve_code_tokens=preserve_ct)
        bow = c_model.changeset_corpus.id2word.doc2bow(issue_text, allow_update=False)
        term_topics_bugs = c_model.lda.get_topics()

        for i in range(0, top_topics):
            topic = bug_topics[i][0]
            f.write('\nTopic {0}: '.format(topic))
            term_topic = term_topics_bugs[topic]
            for id, count in bow:
                f.write('{0} : {1}, '.format(c_model.changeset_corpus.id2word.id2token[id], term_topic[id]))

        f.write('\n################## Bug report tokens #####################')
        for id, count in bow:
            sorted_term_topics = sorted(c_model.lda.get_term_topics(id, 0), key=lambda x: x[1], reverse=True)
            f.write('\n{0} : {1}'.format(c_model.changeset_corpus.id2word.id2token[id], sorted_term_topics))
        f.write('\n#######################################')

        f.write('\n################## Target files #####################')
        for file in target_files:
            for metadata, topics in code_topics:
                if file == metadata[0]:
                    f.write('\n{0}'.format(file))
                    for topic, prob in sorted(list(zip(range(0, len(topics)), topics)), key=lambda x: x[1],
                                              reverse=True)[
                                       :top_topics]:
                        f.write('\nTopic {0} - {1}: {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))
                    break

        f.write('\n################## Recommended files #####################')
        for file in rec_files:
            for metadata, topics in code_topics:
                if file == metadata[0]:
                    f.write('\n{0}'.format(file))
                    for topic, prob in sorted(list(zip(range(0, len(topics)), topics)), key=lambda x: x[1],
                                              reverse=True)[
                                       :top_topics]:
                        f.write('\nTopic {0} - {1}: {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))
                    break


def bug_prediction(project, bug_id, c_model, code_topics, b_model, combined_topics, bug_topics, translated_topics,
                   subranks, all_ranks):
    bug_topics = list(zip(range(0, project.topics[0]), bug_topics.reshape(-1)))
    bug_topics.sort(key=lambda x: x[1], reverse=True)

    combined_topics = list(zip(range(0, project.topics[1]), combined_topics))
    combined_topics.sort(key=lambda x: x[1], reverse=True)

    translated_topics = list(zip(range(0, project.topics[1]), translated_topics.reshape(-1)))
    translated_topics.sort(key=lambda x: x[1], reverse=True)

    results_dir = create_results_dir(project.results_path, 'bug_prediction')
    issue = common.read_issue(project, bug_id)
    max_topics = 5

    preserve_ct = True if project.model_type == 'joined' else False
    with(open(os.path.join(results_dir, '{0}.txt'.format(bug_id)), 'a')) as f:
        f.write('Issue {0}\n'.format(bug_id))
        f.write(issue)

        f.write('\n########### Topics in bug reports model #############')
        for i in range(0, max_topics):
            topic, prob = bug_topics[i]
            f.write('\nTopics {0} - {1} : {2}'.format(topic, prob, b_model.lda.show_topic(topic, topn=5)))

        f.write('\n###########################')
        f.write('\nMost relevant topics to the given word (bug reports model)')
        issue_text = common.preprocess(issue, preserve_code_tokens=preserve_ct)
        bow = b_model.bugs_corpus.id2word.doc2bow(issue_text, allow_update=False)
        for id, count in bow:
            sorted_term_topics = sorted(b_model.lda.get_term_topics(id, minimum_probability=0.0),
                                        key=lambda x: x[1], reverse=True)
            f.write('\n{0} : {1}'.format(b_model.bugs_corpus.id2word.id2token[id], sorted_term_topics))

        f.write('\n###########################')
        f.write('\nTranslated topics:')
        for i in range(0, max_topics):
            topic, prob = translated_topics[i]
            f.write('\nTopics {0} - {1} : {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))

        f.write('\n###########################')
        f.write('\nMost relevant topics to the given word (changesets model)')
        issue_text = common.preprocess(issue, preserve_code_tokens=preserve_ct)
        bow = c_model.changeset_corpus.id2word.doc2bow(issue_text, allow_update=False)
        for id, count in bow:
            sorted_term_topics = sorted(c_model.lda.get_term_topics(id, minimum_probability=0.0),
                                        key=lambda x: x[1], reverse=True)
            f.write('\n{0} : {1}'.format(c_model.changeset_corpus.id2word.id2token[id], sorted_term_topics))

        f.write('\n###########################')
        f.write('\nCombined topics:')
        for i in range(0, max_topics):
            topic, prob = combined_topics[i]
            f.write('\nTopics {0} - {1}: {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))

        target_files = [x[2] for x in subranks]
        rec_files = [x[0] for x in all_ranks[:10]]

        f.write('\n################## Target files #####################')
        for file in target_files:
            for metadata, topics in code_topics:
                if file == metadata[0]:
                    f.write('\n{0}'.format(file))
                    for topic, prob in sorted(list(zip(range(0, len(topics)), topics)), key=lambda x: x[1],
                                              reverse=True)[:5]:
                        f.write('\nTopic {0} - {1}: {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))
                    break

        f.write('\n################## Recommended files #####################')
        for file in rec_files:
            for metadata, topics in code_topics:
                if file == metadata[0]:
                    f.write('\n{0}'.format(file))
                    for topic, prob in sorted(list(zip(range(0, len(topics)), topics)), key=lambda x: x[1],
                                              reverse=True)[:5]:
                        f.write('\nTopic {0} - {1}: {2}'.format(topic, prob, c_model.lda.show_topic(topic, topn=5)))
                    break


def _visualization(project, idx, bug_model, bug_corpus, changeset_model, changeset_corpus):
    results_dir = create_results_dir(project.results_path, idx)
    try:
        _save_model_vis(bug_model, bug_corpus, bug_corpus.id2word, os.path.join(results_dir, 'bugs.html'))
        _save_model_vis(changeset_model, changeset_corpus, changeset_corpus.id2word,
                        os.path.join(results_dir, 'changesets.html'))
    except:
        pass


def _save_model_vis(model, corpus, dict, name):
    plot = pyLDAvis.gensim.prepare(model, corpus, dict)
    pyLDAvis.save_html(plot, name)


def create_results_dir(dir, path):
    utils.mkdir(dir)
    results_dir = os.path.join(dir, path)
    utils.mkdir(results_dir)
    return results_dir


def ranks(dir, issue, issues_subranks, sorted_ranks, doc_topics, BM_ttopics=None):
    results_dir = create_results_dir(dir, "")
    with(open(os.path.join(results_dir, 'ranks'), 'a')) as f:
        f.write("########### {0} ##############\n".format(issue))
        for rel in issues_subranks:
            topics = _top_topics(rel[2], doc_topics)
            f.write('{0}, top topics: {1}\n'.format(rel, topics))
        if BM_ttopics is not None:
            f.write('######## Bug topics ##########\n')
            f.write('{0}\n'.format(
                sorted(list(zip(range(0, len(BM_ttopics)), BM_ttopics)), key=lambda x: x[1], reverse=True)[:5]))
        f.write("### Top 10 ###\n")
        for i in range(0, 10):
            topics = _top_topics(sorted_ranks[i][0], doc_topics)
            f.write('{0}, top topics: {1}\n'.format(sorted_ranks[i], topics))


def _top_topics(doc, doc_topics, n=3):
    for doc_name, topics in doc_topics:
        if doc_name[0] == doc:
            return sorted(list(zip(range(0, len(doc_topics)), topics)), key=lambda x: x[1], reverse=True)[:n]


def metrics_BL(project, ranks, ranks_map, mrr, BD_pred_cnt, C_pred_cnt):
    map = utils.calculate_map(ranks_map)
    top1 = utils.calculate_top(ranks, 1)
    top3 = utils.calculate_top(ranks, 3)
    top5 = utils.calculate_top(ranks, 5)
    top10 = utils.calculate_top(ranks, 10)
    top20 = utils.calculate_top(ranks, 20)

    with (open(os.path.join(project.results_path, 'mrr.txt'), 'w')) as f:
        f.write(str(mrr))
    with (open(os.path.join(project.results_path, 'map.txt'), 'w')) as f:
        f.write(str(map))
    with (open(os.path.join(project.results_path, 'topk.txt'), 'w')) as f:
        f.write('k,accuracyn\n')
        f.write('{0},{1}\n'.format(1, top1))
        f.write('{0},{1}\n'.format(3, top3))
        f.write('{0},{1}\n'.format(5, top5))
        f.write('{0},{1}\n'.format(10, top10))
        f.write('{0},{1}\n'.format(20, top20))

    with (open(os.path.join(project.results_path, 'config.txt'), 'w')) as f:
        f.write(str(project))

    with (open(os.path.join(project.results_path, 'prediction_cnt.txt'), 'w')) as f:
        f.write('joined,{0}\n'.format(BD_pred_cnt))
        f.write('code,{0}'.format(C_pred_cnt))


def mrr_per_bug(path, frms):
    mrrb_fname = os.path.join(path, 'mrr_per_bug.csv')
    if not os.path.exists(mrrb_fname):
        with open(mrrb_fname, 'w') as f:
            f.write('id,mrr\n')

    with open(mrrb_fname, 'a') as f:
        for idx, values in enumerate(frms):
            rank = values[0]
            bug_id = values[1]
            if rank:
                mrr = 1.0 / float(rank)
            else:
                mrr = 0.0
            f.write('{0},{1}\n'.format(bug_id, mrr))


def map_per_bug(path, ranks):
    mapb_fname = os.path.join(path, 'map_per_bug.csv')

    seen = set()
    with open(mapb_fname, 'w') as f:
        f.write('id,map\n')
        for bug_id, recl in ranks:
            vals = list()
            if bug_id in seen:
                continue
            seen.add(bug_id)
            for idx, file_ranks in enumerate(recl):
                rank, _, _ = file_ranks
                vals.append((idx + 1) / float(rank))
            if len(recl) > 0:
                map_value = np.mean(vals)
            else:
                map_value = 0.0
            f.write('{0},{1}\n'.format(bug_id, map_value))

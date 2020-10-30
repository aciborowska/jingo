import sys

import click
import numpy as np
import coloredlogs
import logging
import defaults

import common
import goldsets
import utils

from gensim.matutils import sparse2full

import save
import partition as part
import index
import training_links as tlinks
from translators import TMatrix
from models import ChangesetModel, BugModel

logger = logging.getLogger('main')


@click.command()
@click.option('--verbose', '-v',
              count=True,
              help='Enable verbose output')
@click.option('--name',
              help='Name of project to run the experiment on')
@click.option('--model-type',
              help='Type of model to use',
              type=click.Choice(['changesets', 'joined']),
              default='joined')
@click.option('--datasets',
              help='Path to datasets',
              type=click.Choice(['datasets/corley', 'datasets/bench4bl','datasets/hcc']),
              default='datasets/corley')
@click.option('--topics',
              help='Number of topics for bugs model and changesets model. Format: bugs_k, changes_k.',
              type=(int, int),
              default=[50, 100])
@click.option('--decays',
              help='Decays for bugs model and changesets model. Format bugs_decay, changesets_decay',
              type=(float, float),
              default=[0.75, 1.0])
@click.option('--gamma',
              help='Boosting parameter for joined model prediction',
              type=float,
              default=1.0)
@click.option('--omega',
              help='Boosting number of fixed issues to observe before using joined model',
              type=float,
              default=1.0)
@click.option('--save-model',
              is_flag=True,
              help='Save model after training')
@click.option('--links-limit',
              help='Limit number of links used to build T matrix',
              type=click.Choice(['None', 'min', 'omega']),
              default='None')
@click.option('--save-prediction',
              help='Save prediction of [n] first testing instances',
              type=int)
@click.option('--random-seed-value',
              help='Set the RNG seed value',
              default=defaults.DEFAULT_RND)
def run(verbose, name, *args, **kwargs):
    setup(verbose, kwargs)

    config = defaults.setup_default_config(kwargs)
    projects = read_projects(name, config)

    for project in projects:
        run_experiment(project)


def run_experiment(project):
    logger.info('Running project on %s', str(project))
    utils.mkdir(project.save_model_path)
    utils.mkdir(project.results_path)

    repos = common.load_repos(project)
    goldset = goldsets.create_goldsets(project, ids=common.load_fixed_ids(project))

    logger.info('Creating models')
    changesets_model = ChangesetModel(project, repos)
    bugs_model = BugModel(project, repos)

    mtranslator = TMatrix(project)

    logger.info('Partitioning corpus')
    links = part.generate_links(project, changesets_model, bugs_model)
    all_fixed_links = part.generate_links(project, changesets_model, bugs_model, version_ids=False)

    training_links = tlinks.Links(all_fixed_links, project, changesets_model, bugs_model)
    ranks = dict()
    ranks_map = list()
    classes = index.list_source_code_files(project)

    BD_pred_cnt = 0
    C_pred_cnt = 0
    for idx, link in enumerate(links):
        logger.info('Processing {0}/{1} partitions'.format(idx, len(links)))

        changesets_model.update(link)
        if project.model_type == 'joined':
            bugs_model.update(link)

        _, _, _, sha, _, _, _, fixed_bugs = link

        for bug_id in fixed_bugs:
            if bug_id not in goldset:
                logger.info('Prediction for bug {0} skipped - no goldset.'.format(bug_id))
                continue

            logger.info("Predict files for bug {0}".format(bug_id))

            if project.model_type == 'joined':
                # first, check if there is enough data to train translator
                timestamp = changesets_model.changeset_ts[sha]
                training_links.update_available_links(link, timestamp)
                A, B = training_links.get_links()

                if mtranslator.is_trainable(A):
                    topics_dist, bug_topics, translated_topics = joined_prediction(mtranslator, bug_id, classes,
                                                                                   changesets_model, bugs_model,
                                                                                   project, A, B)
                    BD_pred_cnt += 1

            if (project.model_type == 'joined' and not mtranslator.trainable) or project.model_type == 'changesets':
                topics_dist = changeset_prediction(changesets_model.lda, bugs_model.bugs_corpus, bug_id)
                C_pred_cnt += 1

            other_corpus = common.create_release_corpus(project, repos, ref=sha, mallet=False)

            query_topics = [((bug_id, project.name), topics_dist)]
            doc_topic = common.get_topics(changesets_model.lda, other_corpus)
            subranks, sorted_ranks = common.get_rank(project, query_topics, doc_topic, goldset)

            if bug_id in subranks:
                if bug_id not in ranks:
                    ranks[bug_id] = list()

                rank = subranks[bug_id]
                ranks[bug_id].extend(rank)
                ranks_map.append((bug_id, rank))
            else:
                logger.info('Couldnt find qid %s', bug_id)

            # save some data to analyze later
            if project.save_prediction:
                if project.model_type == 'joined' and mtranslator.trainable:
                    save.bug_prediction(project, bug_id, changesets_model, doc_topic, bugs_model, topics_dist,
                                        bug_topics, translated_topics, subranks[bug_id], sorted_ranks)
                else:
                    save.changeset_prediction(project, bug_id, changesets_model, doc_topic, topics_dist,
                                              subranks[bug_id], sorted_ranks)

            save.ranks(project.results_path, bug_id, subranks[bug_id], sorted_ranks, doc_topic)

    frms = common.get_frms(ranks, goldset)
    mrr = utils.calculate_mrr(num for num, _, _ in frms)
    save.mrr_per_bug(project.results_path, frms)
    save.map_per_bug(project.results_path, ranks_map)

    common.write_ranks(project, 'all', ranks, path=project.results_path)
    save.metrics_BL(project, ranks, ranks_map, mrr, BD_pred_cnt, C_pred_cnt)


def joined_prediction(mtranslator, bug_id, classes, c_model, b_model, project, A, B):
    mtranslator.fit(A, B)

    issue_bow = common.issue2bow(project, b_model.bugs_corpus, bug_id)
    BM_topics = sparse2full(b_model.lda[issue_bow], project.topics[0])

    translated_topics = mtranslator.predict(BM_topics)

    CM_topics = changeset_prediction(c_model.lda, b_model.bugs_corpus, bug_id)
    words_no, tokens_no = index.count(project, bug_id, classes)
    ratio = words_no / float(tokens_no + words_no)
    combined_topics = utils.norm((translated_topics * ratio) + (((1.0 - ratio) * project.gamma) * CM_topics))
    return combined_topics, BM_topics, translated_topics


def changeset_prediction(model, queries, qid):
    query_topic = common.get_topics(model, queries, by_ids=[qid])
    return query_topic[0][1]


def read_projects(name, params):
    projects = common.load_projects(params, path=params['datasets'])

    if name:
        name = name.lower()
        projects = [x for x in projects if x.name == name]

    return projects


def setup(verbose, params):
    random_seed_value = params['random_seed_value']
    np.random.seed(random_seed_value)

    coloredlogs.install()

    if verbose > 1:
        coloredlogs.set_level(logging.DEBUG)
    elif verbose == 1:
        coloredlogs.set_level(logging.INFO)


if __name__ == '__main__':
    run(sys.argv[1:])

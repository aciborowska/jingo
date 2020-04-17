import os
import logging

from gensim.models import LdaModel, LdaMulticore

import corpora
import common
import defaults

logger = logging.getLogger('models')


class Model(object):

    def __init__(self, project):
        self.project = project

    def build_model(self, id2word, corpus=None):
        if corpus is None:
            p = defaults.onlineLDA_update_settings(self.project)
        else:
            p = self.project

        return self.create_model(p, id2word, corpus)

    def create_model(self, project, id2word, corpus):
        model_fname = self.get_model_fname(project)
        model_save_path = os.path.join(project.save_model_path, model_fname)

        if not os.path.exists(model_save_path):
            model = self.model(project, corpus, id2word)

            if project.save_model:
                logger.info('Saving model to file {0}'.format(model_save_path))
                model.save(model_save_path)
                logger.info('Model saved')
        else:
            logger.info('Loading model from file {0}'.format(model_save_path))
            model = LdaModel.load(model_save_path)
            logger.info('Model loaded')

        return model, model_save_path

    def model(self, project, corpus, id2word):
        model_config, model_config_string = self.get_model_config(project)
        if type == 'bugs':
            model_config = project.bugs_model_config
            model_config_string = project.bugs_model_config_string
        elif type == 'changes':
            model_config = project.changes_model_config
            model_config_string = project.changes_model_config_string

        params = dict(model_config)
        params['corpus'] = corpus
        params['id2word'] = id2word

        logger.info('Building model for project {0} for {1} with model parameters: {2}; and {3}'.
                    format(project.name, type, model_config_string, project.changeset_config_string))

        model = LdaModel(**params)
        return model


class ChangesetModel(Model):

    def __init__(self, project, repos):
        super(ChangesetModel, self).__init__(project)
        self.changeset_corpus, self.changeset_ts, self.commit_log = self.build_corpus(repos)
        self.lda, self.model_fname = self.build_model(self.changeset_corpus.id2word)

    def build_corpus(self, repos):
        changeset_corpus, changeset_ts, commit_log = common.create_corpus(self.project, repos, corpora.ChangesetCorpus,
                                                                          use_level=False)
        return changeset_corpus, changeset_ts, commit_log

    def get_model_fname(self, project):
        return project.changes_model_config_string + '.changesets.gz'

    def get_model_config(self, project):
        return project.changes_model_config, project.changes_model_config_string

    def update(self, link):
        # (prev_commit_idx, idx_fix_start, idx_fix_end, sha, commit_ts, prev_bug_idx, end_bug_idx, fixed_bugs)
        prev_commit_idx, idx_fix_start, _, sha, _, _, _, _ = link
        sha_model_fname = self.model_fname % sha

        if os.path.exists(sha_model_fname):
            self.lda = self.lda.load(sha_model_fname)
        else:
            logger.info('Update changeset model')
            docs = self.changeset_corpus[prev_commit_idx:idx_fix_start]

            logger.info('Processing chunk with {0} entries'.format(len(docs)))
            self.lda.update(docs, len(docs))

            if self.project.save_model:
                self.lda.save(sha_model_fname)


class BugModel(Model):

    def __init__(self, project, repos):
        super(BugModel, self).__init__(project)
        self.commits_corpus, self.bugs_corpus = self.build_corpus(repos)
        self.lda, self.model_fname = self.build_model(self.bugs_corpus.id2word)

    def build_corpus(self, repos):
        bug_corpus = common.create_queries(self.project, id2word=None)
        return None, bug_corpus

    def get_model_fname(self, project):
        return project.bugs_model_config_string + '.bugs.gz'

    def get_model_config(self, project):
        return project.bugs_model_config, project.bugs_model_config_string

    def update(self, link):
        # (prev_commit_idx, idx_fix_start, idx_fix_end, sha, commit_ts, prev_bug_idx, end_bug_idx, fixed_bugs)
        prev_commit_idx, idx_fix_start, _, sha, _, prev_bug_idx, end_bug_idx, _ = link
        sha_model_fname = self.model_fname % sha

        if os.path.exists(sha_model_fname):
            self.lda = self.lda.load(sha_model_fname)
        else:
            docs = self.bugs_corpus[prev_bug_idx: end_bug_idx]

            if end_bug_idx >= prev_bug_idx and len(docs) > 0:
                logger.info('Update bug model.\nProcessing chunk with {0} entries'.format(len(docs)))
                self.lda.update(docs, len(docs))

            if self.commits_corpus is not None:
                docs = self.commits_corpus[prev_commit_idx:idx_fix_start]
                self.lda.update(docs, len(docs))

        if self.project.save_model:
            self.lda.save(sha_model_fname)

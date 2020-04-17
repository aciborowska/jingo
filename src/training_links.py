import common
from gensim.matutils import sparse2full
import partition as part


class Links:

    def __init__(self, links, project, changesets_model, bugs_model):
        self.changesets = []
        self.bugs = []
        self.changesets_model = changesets_model
        self.bugs_model = bugs_model
        self.all_fixed_links = links
        self.project = project
        self.last_commit_idx = 0
        self.seen_fixes = set()
        self.preserve_ct = True if project.model_type == 'joined' else False

    def update_available_links(self, current_link, timestamp):
        #commit logs
        self.commitlog2matrix(current_link)

        #bug fixes
        bugfix_links = part.generate_links_by_timestamps(self.all_fixed_links, timestamp)
        self.links2matrix(bugfix_links)

        assert len(self.bugs) == len(self.changesets)
        self.limit_links(self.bugs)
        self.limit_links(self.changesets)

        self.last_commit_idx = current_link[1]

    def limit_links(self, links):
        if self.project.links_limit == 'None':
            return links
        elif self.project.links_limit == 'min':
            return links[-max(self.project.topics):]
        elif self.project.links_limit == 'omega':
            return links[-int(max(self.project.topics) * self.project.omega):]
        else:
            raise ValueError(
                'Uknown option {0} for --links-limit. Use: [\'None\', \'min\', \'omega\']'.format(
                    self.project.links_limit))

    def links2matrix(self, links):
        # link format:
        # (prev_commit_idx, idx_fix_start, idx_fix_end, sha, commit_ts, prev_bug_idx, end_bug_idx, fixed_bugs)

        for link in links:
            _, idx_fix_start, _, _, _, _, _, fixed_bugs = link
            if idx_fix_start in self.seen_fixes:
                continue

            self.seen_fixes.add(idx_fix_start)
            for issue in fixed_bugs:
                doc = self.changesets_model.changeset_corpus[idx_fix_start]
                self.changesets.append(doc)
                bug = get_issue_bow(self.bugs_model.bugs_corpus, self.project, issue)
                self.bugs.append(bug)

    def commitlog2matrix(self, link):
        # link format:
        # (prev_commit_idx, idx_fix_start, idx_fix_end, sha, commit_ts, prev_bug_idx, end_bug_idx, fixed_bugs)
        self.changesets_model.changeset_corpus.metadata = True
        last_commit_idx = link[1]
        for idx in range(self.last_commit_idx, last_commit_idx):
            changeset, metadata = self.changesets_model.changeset_corpus[idx]
            log = self.changesets_model.commit_log[metadata[0]]
            log = preprocess_log(log, self.preserve_ct)
            log_bow = self.bugs_model.bugs_corpus.id2word.doc2bow(log, allow_update=False)
            if len(log_bow) > 5:
                self.changesets.append(changeset)
                self.bugs.append(log_bow)

        self.changesets_model.changeset_corpus.metadata = False

    def get_links(self):
        A = []
        B = []

        for changeset in self.changesets:
            changeset_topics = sparse2full(self.changesets_model.lda[changeset],
                                           self.project.changes_model_config['num_topics'])
            A.append(changeset_topics)

        for bug in self.bugs:
            bug_topics = sparse2full(self.bugs_model.lda[bug], self.project.bugs_model_config['num_topics'])
            B.append(bug_topics)

        return A, B


def preprocess_log(text, preserve_code_tokens):
    text = text.split(' ')
    text = ' '.join([x for x in text if not x.startswith('git-svn-id') and not x.startswith('https://')])
    return common.preprocess(text, preserve_code_tokens=preserve_code_tokens)


def get_issue_bow(bug_corpus, project, issue):
    preserve_ct = True if project.model_type == 'joined' else False
    issue_text = common.read_issue(project, issue)
    issue_text = common.preprocess(issue_text, preserve_code_tokens=preserve_ct)
    issue_bow = bug_corpus.id2word.doc2bow(issue_text, allow_update=False)
    return issue_bow

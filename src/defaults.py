# adapted from https://github.com/cscorley/triage
DEFAULT_RND = 1234


def setup_default_config(kwargs):
    bugs_k = kwargs['topics'][0]
    changes_k = kwargs['topics'][1]
    bugs_decay = kwargs['decays'][0]
    changes_decay = kwargs['decays'][1]

    changeset_config, changeset_config_string = changesets_config(kwargs)
    kwargs.update({'changeset_config': changeset_config,
                   'changeset_config_string': changeset_config_string})

    logs_config, logs_config_string = commit_logs_config()
    kwargs.update({'commit_logs_config': logs_config,
                   'commit_logs_config_string': logs_config_string})

    bugs_config, bugs_config_string = model_config(kwargs, bugs_k, bugs_decay)
    kwargs.update({'bugs_model_config': bugs_config,
                   'bugs_model_config_string': bugs_config_string})

    changes_config, changes_config_string = model_config(kwargs, changes_k, changes_decay)
    kwargs.update({'changes_model_config': changes_config,
                   'changes_model_config_string': changes_config_string})

    return kwargs


def model_config(kwargs, num_topics=20, decay=0.5):
    if "random_seed_value" in kwargs:
        random_seed_value = kwargs["random_seed_value"]
    else:
        random_seed_value = DEFAULT_RND

    config = {
        'num_topics': num_topics,
        'alpha': 0.2,
        'eta': 0.2,
        'decay': decay,  # between (0.5, 1], controls forgetting lambda when new docs arrives
        'offset': 1.0,  # > 0, downweights early iterations
        'iterations': 1000,
        'passes': 10,
        'random_state': random_seed_value
        # 'minimum_probability': 0.0
    }

    config_str = '-'.join([str(v) for k, v, in sorted(config.items())])
    return config, config_str


def changesets_config(kwargs):
    if kwargs['model_type'] == 'changesets':
        include_filenames = False
    else:
        include_filenames = True
    config = {
        'include_additions': True,
        'include_context': True,
        'include_message': False,
        'include_removals': True,
        'include_filenames': include_filenames,
        'divide_commits': False
    }

    config_str = '-'.join([str(v) for k, v, in sorted(config.items())])
    return config, config_str


def onlineLDA_update_settings(project):
    project = onlineLDA_update_bugs(project)
    project = onlineLDA_update_changes(project)
    return project


def onlineLDA_update_bugs(project):
    project.bugs_model_config.update({
        'alpha': 'auto',
        'eta': 'auto',
        'chunksize': 1,
        'passes': 10,
        'offset': 1,
        'eval_every': 0,
    })
    p = project._replace(
        bugs_model_config_string='-'.join(
            ['%s'] + [str(v) for k, v in sorted(project.bugs_model_config.items())]))
    return p


def onlineLDA_update_changes(project):
    project.changes_model_config.update({
        'alpha': 'auto',
        'eta': 'auto',
        'chunksize': 1,
        'passes': 10,
        'offset': 1,
        'eval_every': 0,
    })
    p = project._replace(
        changes_model_config_string='-'.join(
            ['%s'] + [str(v) for k, v in sorted(project.changes_model_config.items())]))

    return p


def commit_logs_config():
    config = {
        'divide_commits': False
    }
    config_str = '-'.join([str(v) for k, v, in sorted(config.items())])
    return config, config_str

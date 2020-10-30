# from https://github.com/cscorley/triage
import io
from subprocess import Popen, PIPE
from xml.sax.saxutils import unescape
import codecs
import errno
import os.path
import time

from smart_open import smart_open, open
from lxml import etree
import dulwich
import dulwich.objects
import dulwich.diff_tree
import dulwich.patch
import re
import requests
import whatthepatch as wtp

from common import *
import utils as utils

logger = logging.getLogger('goldsets')

os.environ['TZ'] = 'UTC'


def create_goldsets(project, ids=None):
    logger.info("Loading goldsets for project: %s", str(project))

    if ids is None:
        ids = load_ids(project)
    issue2git, git2issue = load_issue2git(project, ids)

    commit_golds = load_goldset(project, level='file')

    goldsets = dict()
    for id_ in ids:
        if id_ in issue2git:
            shas = issue2git[id_]
            for sha in shas:
                if sha in commit_golds:
                    commit, changes = commit_golds[sha]
                    if id_ not in goldsets:
                        goldsets[id_] = set()

                    goldsets[id_].update([str(item) for kind, item in changes if item.endswith(".java")])

    logger.info("Returning %d goldsets", len(goldsets))
    return goldsets


def get_refs(repo):
    refs = repo.get_refs()
    ref2commit = dict()
    if 'HEAD' in refs:
        ref2commit['HEAD'] = refs['HEAD']

    ref2commit.update(dict((ref, sha)
                           for ref, sha in refs.items()
                           if (ref.startswith('refs/tags/'))))

    ref2commit.update(dict((ref, sha)
                           for ref, sha in refs.items()
                           if (ref.startswith('refs/heads/'))))

    return ref2commit


def load_goldset(project, level='file'):
    goldset = dict()
    with open(os.path.join(project.data_path, 'changes-%s.log.gz' % level)) as f:
        for commit, changes in commit_parser(f):
            goldset[commit.id.decode('utf-8')] = (commit, changes)

    return goldset


def commit_parser(f):
    commit = None
    text = list()
    changes = list()

    while True:
        line = next(f).strip().split()
        if commit is None and line and line[0] == 'commit':
            commit = line[0:2]
            text = list()
            text.append(' '.join(commit) + '\n')

        if commit:
            line = next(f)
            while True:
                ls = line.split()

                if ls and line[0] in ['M', 'D', 'A']:
                    break
                else:
                    line_data = line.strip().split(None, 1)
                    if len(line_data) == 2 and line_data[0] == 'commit':
                        line_data = line.strip().split()[0:2]
                        if dulwich.objects.valid_hexsha(line_data[1]):
                            commit = line_data
                            text = list()
                            text.append(' '.join(commit) + '\n')
                            line = next(f)
                            continue

                text.append(line)
                line = next(f)

            changes = list()
            line = line.strip().split(None, 1)
            while line:
                changes.append(line)
                line = next(f).strip().split(None, 1)

            yield (dulwich.objects.ShaFile.from_raw_string(1,  # 1 is 'Commit'
                                                           ''.join(text).encode('utf-8'),
                                                           sha=commit[1]), changes)
            commit = None


def build_goldset(project, path='corley/data'):
    if os.path.exists(os.path.join(project.data_path, 'DONE')):
        return

    repos = load_repos(project, path)
    issue2git = list()

    matcher = re.compile(r"%s-(\d+)" % project.name.upper())

    filelog = smart_open(os.path.join(project.data_path, 'changes-file.log.gz'), 'wb')
    classlog = smart_open(os.path.join(project.data_path, 'changes-class.log.gz'), 'wb')
    methodlog = smart_open(os.path.join(project.data_path, 'changes-method.log.gz'), 'wb')

    for repo_num, repo in enumerate(repos):
        ref2commit = get_refs(repo)
        commit2ref = dict((v, k) for k, v in ref2commit.items())

        ref = None
        prev = (None, None)
        changes = list()
        for commit, parent, change in walk_changes(project, repo, commit2ref.keys()):
            sha = commit.id
            links = list(matcher.findall(commit.message))
            if prev != (sha, parent):
                # on new commit, dump changes
                write_logs(filelog, classlog, methodlog, repo, commit, changes)

                prev = (sha, parent)
                changes = list()

            if change:
                changes.append(change)

            if sha in commit2ref:
                ref = commit2ref[sha]

            for issue in links:
                issue2git.append((issue, repo_num, sha))

        # write out the last set of changes
        write_logs(filelog, classlog, methodlog, repo, commit, changes)

    filelog.close()
    methodlog.close()
    classlog.close()

    utils.mkdir(os.path.join(project.data_path))

    with open(os.path.join(project.data_path, 'issue2git.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow("issue repo sha".split())
        writer.writerows(issue2git)

    download_jira_bugs(project)

    with open(os.path.join(project.data_path, 'DONE'), 'w') as f:
        f.write('yes')


def make_set(added, removed):
    added = set(dulwich.objects.TreeEntry(x.full_name, None, None) for x in added)
    removed = set(dulwich.objects.TreeEntry(x.full_name, None, None) for x in removed)

    items = set()
    modified = removed & added
    added = added - modified
    removed = removed - modified

    for each in removed:
        t = dulwich.diff_tree.TreeChange('deleted', each, each)
        items.add(t)
    for each in added:
        t = dulwich.diff_tree.TreeChange('added', each, each)
        items.add(t)
    for each in modified:
        t = dulwich.diff_tree.TreeChange('modified', each, each)
        items.add(t)

    return items


def write_logs(filelog, classlog, methodlog, repo, commit, changes):
    write_log(filelog, commit, changes)

    methods = set()
    classes = set()
    for mremoved, madded, cremoved, cadded in parse_changes(repo, changes):
        methods.update(make_set(mremoved, madded))
        classes.update(make_set(cremoved, cadded))

    write_log(methodlog, commit, methods)
    write_log(classlog, commit, classes)


def write_log(f, commit, changes):
    for change in changes:
        f.write(change.type.upper()[0] + '      ')
        path = '/dev/null'
        if change.old.path and change.new.path:
            path = change.new.path
        elif change.old.path:
            path = change.old.path
        elif change.new.path:
            path = change.new.path

        f.write(to_utf8(path.strip().rstrip() + '\n'))

    if changes:
        f.write('\n')

    f.write('commit %s\n' % commit.id)
    raw_lines = commit.as_raw_string().splitlines()

    # tab the message like git
    tab = False
    for line in raw_lines:
        if not tab and not line:
            tab = True

        if tab and line:
            f.write('    ')

        if line:
            f.write(to_utf8(line))

        f.write('\n')

    f.write('\n')


def download_jira_bugs(project):
    url_base = 'https://issues.apache.org/jira/si/jira.issueviews:issue-xml/%s/%s.xml'
    path = os.path.join(project.data_path, 'queries')
    xmlpath = os.path.join(project.data_path, 'issues.xml.gz')
    utils.mkdir(path)
    utils.mkdir(os.path.join(path, 'short'))
    utils.mkdir(os.path.join(path, 'long'))

    p = etree.XMLParser()
    hp = etree.HTMLParser()

    downloaded = set()
    fixedin = dict()
    affected = dict()

    with smart_open(xmlpath, 'wb') as xmlfile:
        xmlfile.write("<jira>\n")

        bugid = 0
        fail_attempts_cnt = 0
        while fail_attempts_cnt < 50:
            bugid += 1
            logger.info("Fetching bugid %s", bugid)
            fname = project.name.upper() + '-' + str(bugid)

            r = try_request(url_base % (fname, fname))
            r = to_unicode(r.text)

            xmlfile.write(to_utf8(r))

            try:
                tree = etree.parse(io.StringIO(r), p)
            except etree.XMLSyntaxError:
                logger.error("Error in XML: %s %s %s", bugid, project, project.version)
                fail_attempts_cnt += 1
                continue

            root = tree.getroot()

            type = root.find('channel').find('item').find('type').text
            if type != "Bug":
                continue

            html = root.find('channel').find('item').find('description').text
            summary = root.find('channel').find('item').find('summary').text
            summary = to_unicode(summary)

            fixversion = ["v" + x.text for x in root.findall('.//fixVersion')]
            affected_versions = ["v" + x.text for x in root.find('channel').find('item').findall('.//version')]

            time = root.find('channel').find('item').findall('.//created')
            assert len(time) == 1
            created_time = get_time(time[0].text)

            htree = etree.parse(io.StringIO(html), hp)
            desc = ''.join(htree.getroot().itertext())
            desc = to_unicode(desc)

            with codecs.open(os.path.join(path, 'short', '%s.txt' % bugid), 'w', 'utf-8') as f:
                f.write(summary)

            with codecs.open(os.path.join(path, 'long', '%s.txt' % bugid), 'w', 'utf-8') as f:
                f.write(desc)

            downloaded.add((bugid, created_time))

            for ver in fixversion:
                if ver not in fixedin:
                    fixedin[ver] = list()
                fixedin[ver].append(str(bugid))

            for ver in affected_versions:
                if ver not in affected:
                    affected[ver] = list()
                affected[ver].append((bugid, created_time))

        xmlfile.write("</jira>\n")

    for ver, bugs in fixedin.items():
        utils.mkdir(os.path.join(project.data_path, ver))
        bugs_sorted = sorted(bugs)
        with open(os.path.join(project.data_path, ver, 'fixed_ids.txt'), 'w') as f:
            f.write('\n'.join(bugs_sorted))

    for ver, bugs in affected.items():
        utils.mkdir(os.path.join(project.data_path, ver))
        save_bugs(os.path.join(project.data_path, ver, 'affected_ids.txt'), bugs)

    save_bugs(os.path.join(project.data_path, 'bug_ids.txt'), downloaded)

    return sorted(map(int, [x[0] for x in downloaded]))


def save_bugs(path, bugs_times):
    bugs_sorted = sorted(bugs_times, key=lambda x: x[1])
    with open(path, 'w') as f:
        for bug, time in bugs_sorted:
            f.write('{0},{1}\n'.format(bug, time))


def get_time(date_str):
    date_str = date_str.split(',')[1].strip()
    data_format = '%d %b %Y %H:%M:%S +0000'
    date = time.strptime(date_str, data_format)
    # to epoch time - easier to compare
    return int(time.mktime(date))


def try_request(url, n=10):
    try:
        return requests.get(url)
    except:
        time.sleep(600 / n)
        try_request(url, n - 1)


def parse_changes(repo, changes):
    for change in changes:
        for diff in wtp.parse_patch(get_diff(repo, change)):
            removed = list()
            added = list()
            if not (diff.header.old_path.endswith(".java") or
                    diff.header.new_path.endswith(".java")):
                continue

            for r, a, text in diff.changes:
                if r:
                    removed.append(r)
                if a:
                    added.append(a)

            mremoved_blocks = set()
            cremoved_blocks = set()
            if diff.header.old_path != "/dev/null":
                logger.info("Generating XML for file %s @ %s", change.old.path, change.old.sha)
                ftext = repo[change.old.sha].as_raw_string()
                mremoved_blocks, cremoved_blocks = get_blocks(ftext, removed)

            madded_blocks = set()
            cadded_blocks = set()
            if diff.header.new_path != "/dev/null":
                logger.info("Generating XML for file %s @ %s", change.new.path, change.new.sha)
                ftext = repo[change.new.sha].as_raw_string()
                madded_blocks, cadded_blocks = get_blocks(ftext, added)

            yield mremoved_blocks, madded_blocks, cremoved_blocks, cadded_blocks


def walk_changes(project, repo, includes):
    """ Returns one file change at a time, not the entire diff.

    """

    for walk_entry in repo.get_walker(include=includes):
        commit = walk_entry.commit

        # initial revision, has no parent
        if len(commit.parents) == 0:
            changes = dulwich.diff_tree.tree_changes(repo.object_store,
                                                     None,
                                                     commit.tree)
            if changes:
                for change in changes:
                    yield commit, None, change
            else:
                yield commit, None, None

        for parent in commit.parents:
            # do I need to know the parent id?

            changes = dulwich.diff_tree.tree_changes(repo.object_store,
                                                     repo[parent].tree,
                                                     commit.tree)
            if changes:
                for change in changes:
                    yield commit, parent, change
            else:
                yield commit, parent, None


def get_diff(repo, changeset):
    patch_file = io.StringIO()
    dulwich.patch.write_object_diff(patch_file,
                                    repo.object_store,
                                    changeset.old, changeset.new)
    return patch_file.getvalue()


def get_blocks(text, line_nums):
    cmd = "java -cp ./lib -jar ./lib/srcMLOLOL.jar Java".split()
    xml = pipe(text, cmd)
    xml = xml[len('<?xml version="1.0" encoding="UTF-8" standalone="no"?>'):]

    # while here, we could build the text, too?

    # need huge_tree since the depth gets kinda crazy
    p = etree.XMLParser(huge_tree=True)
    try:
        tree = etree.fromstring(xml, parser=p)
    except etree.XMLSyntaxError:
        return set(), set()

    package_name = ''
    pkg = list()
    package = tree.find(".//PackageDeclaration")
    if package is not None:
        qn = package.find(".//QualifiedName")
        for child in qn:
            if child.tag == "CommonToken" and child.attrib["name"] == "Identifier":
                pkg.append(child.text)
        package_name = '.'.join(pkg)

    classTypes = ['ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationTypeDeclaration']

    # these are the method-like decls that come from the 4 'types' in the grammar
    methodTypes = ["MethodDeclaration", "GenericMethodDeclaration",
                   "ConstructorDeclaration", "GenericConstructorDeclaration",
                   "InterfaceMethodDeclaration", "InterfaceGenericMethodDeclaration",
                   "AnnotationMethodRest"]

    mblocks = list()
    cblocks = list()

    for level in ['class', 'method']:
        if level == 'class':
            findtype = classTypes
        elif level == 'method':
            findtype = methodTypes

        for t in findtype:
            for elem in tree.iterfind(".//" + t):
                # in all types, there is a keyword marking the type
                # with the identifier following immediately
                params = list()
                elem_name = None
                for child in elem:
                    if elem_name is None and child.tag == "CommonToken" and child.attrib["name"] == "Identifier":
                        elem_name = child.text

                    if child.tag == "FormalParameters":
                        for param in child.findall(".//Type"):
                            params.append(''.join(ct.text for ct in param.findall(".//CommonToken")))

                if elem_name is None:
                    elem_name = "$" + t + "$"

                start_line = int(elem.attrib["start_line"])
                end_line = int(elem.attrib["end_line"])

                # TODO, find the body, get actual start_line
                body_line = start_line

                supers = list()
                parent = elem.getparent()
                while parent != None:
                    if parent.tag == 'classCreatorRest':
                        # anonymous class time!
                        supers.append("$classCreatorRest$")
                    else:
                        for child in parent:
                            if child.tag == "CommonToken" and child.attrib["name"] == "Identifier":
                                supers.append(child.text)
                                break
                    parent = parent.getparent()

                if package_name:
                    supers.append(package_name)

                if level == 'method':
                    elem_name += '(' + ','.join(params) + ')'

                elem_name = unescape(elem_name)
                supers = [unescape(s) for s in supers]

                block = Block(t, elem_name, start_line, body_line, end_line,
                              super_block_name=u'.'.join(reversed(supers)))

                if level == 'method':
                    mblocks.append(block)
                elif level == 'class':
                    cblocks.append(block)

    # figure out which methods changed
    mchanged = list()
    for method in mblocks:
        keep = False
        for line_num in line_nums:
            if line_num in method.line_range:
                keep = True
        if keep:
            mchanged.append(method)

    cchanged = list()
    for class_ in cblocks:
        keep = False
        for line_num in line_nums:
            if line_num in class_.line_range:
                keep = True
        if keep:
            cchanged.append(class_)

    # for c in changed:
    #    print(c.full_name)

    # shits the bed on unicode, but above works lol
    # print(changed)

    return set(mchanged), set(cchanged)


def pipe(text, cmd):
    p = Popen(cmd, stdin=PIPE, stdout=PIPE)
    try:
        p.stdin.write(text)
    except IOError as e:
        if e.errno == errno.EPIPE or e.errno == errno.EINVAL:
            # Stop loop on "Invalid pipe" or "Invalid argument".
            # No sense in continuing with broken pipe.
            pass
        else:
            # Raise any other error.
            raise

    p.stdin.close()
    return p.stdout.read().decode('UTF-8')

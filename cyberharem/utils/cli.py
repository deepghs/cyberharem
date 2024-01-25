import types

import click
from click.core import Context, Option

from ..config.meta import __TITLE__, __VERSION__, __AUTHOR__, __AUTHOR_EMAIL__

_raw_authors = [item.strip() for item in __AUTHOR__.split(',') if item.strip()]
_raw_emails = [item.strip() for item in __AUTHOR_EMAIL__.split(',')]
if len(_raw_emails) < len(_raw_authors):  # pragma: no cover
    _raw_emails += [None] * (len(_raw_authors) - len(_raw_emails))
elif len(_raw_emails) > len(_raw_authors):  # pragma: no cover
    _raw_emails[len(_raw_authors) - 1] = tuple(_raw_emails[len(_raw_authors) - 1:])
    del _raw_emails[len(_raw_authors):]

_author_tuples = [
    (author, tuple([item for item in (email if isinstance(email, tuple) else ((email,) if email else ())) if item]))
    for author, email in zip(_raw_authors, _raw_emails)
]
_authors = [
    author if not emails else '{author} ({emails})'.format(author=author, emails=', '.join(emails))
    for author, emails in _author_tuples
]

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


def print_version(module, ctx: Context, param: Option, value: bool) -> None:
    """
    Print version information of cli
    :param module: current module using this cli.
    :param ctx: click context
    :param param: current parameter's metadata
    :param value: value of current parameter
    """
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover

    if module is None:
        title = __TITLE__
    elif isinstance(module, types.ModuleType):
        title = module.__name__
    else:
        title = str(module)

    click.echo('{title}, version {version}.'.format(title=title.capitalize(), version=__VERSION__))
    if _authors:
        click.echo('Developed by {authors}.'.format(authors=', '.join(_authors)))
    ctx.exit()

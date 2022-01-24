# coding=utf-8

import six

import pybtex.database
import pybtex.io

from pybtex.backends.html import Backend as BaseBackend
from pybtex.style.formatting.plain import Style as PlainStyle


class HTMLBackend(BaseBackend):
    def __init__(self, *args, **kwargs):
        super(HTMLBackend, self).__init__(*args, **kwargs)
        self._html = ''

    def output(self, html):
        self._html += html

    def format_protected(self, text):
        if text[:4] == 'http':
            return self.format_href(text, text)
        else:
            return r'<span class="bibtex-protected">{}</span>'.format(text)

    def write_prologue(self):
        self.output('<div class="bibliography"><dl>')

    def write_epilogue(self):
        self.output('</dl></div>')

    def _repr_html(self, formatted_bibliography):
        self.write_prologue()
        for entry in formatted_bibliography:
            self.write_entry(entry.key, entry.label, entry.text.render(self))
        self.write_epilogue()

        return self._html.replace('\n', ' ').replace('\\url <a', '<a')


# FIXME: For python 3.7+ only...
# from functools import singledispatch
# from collections.abc import Iterable
#
# # noinspection PyUnusedLocal
# @singledispatch
# def bib2html(bib, style=None, backend=None):
#     raise NotImplementedError('I do not now how to convert a {} type to a bibliography'.format(type(bib)))
def bib2html(bib, style=None, backend=None):
    if isinstance(bib, six.string_types):
        return _bib2html_string(bib, style=style, backend=backend)
    if isinstance(bib, (list, set, tuple)):
        return _bib2html_list(bib, style=style, backend=backend)
    if isinstance(bib, pybtex.database.BibliographyData):
        return _bib2html_bibdata(bib, style=style, backend=backend)
    else:
        raise NotImplementedError('I do not now how to convert a {} type to a bibliography'.format(type(bib)))


# FIXME: For python 3.7+ only...
# @bib2html.register
# def _bib2html_string(bib: str, style=None, backend=None):
def _bib2html_string(bib, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    formatted_bib = style.format_bibliography(pybtex.database.parse_file(bib))

    return backend._repr_html(formatted_bib)


# FIXME: For python 3.7+ only...
# @bib2html.register
# def _bib2html_list(bib: Iterable, style=None, backend=None):
def _bib2html_list(bib, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    bibliography = pybtex.database.BibliographyData()
    for bib_file in bib:
        temp_bib = pybtex.database.parse_file(bib_file)
        for key, entry in temp_bib.entries.items():
            try:
                bibliography.add_entry(key, entry)
            except pybtex.database.BibliographyDataError:
                continue

    formatted_bib = style.format_bibliography(bibliography)

    return backend._repr_html(formatted_bib)


# FIXME: For python 3.7+ only...
# @bib2html.register
# def _bib2html_bibdata(bib: pybtex.database.BibliographyData, style=None, backend=None):
def _bib2html_bibdata(bib, style=None, backend=None):
    if style is None:
        style = PlainStyle()
    if backend is None:
        backend = HTMLBackend()

    formatted_bib = style.format_bibliography(bib)

    return backend._repr_html(formatted_bib)
